#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arpa/inet.h>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>

#define RX_RING_SIZE 1024
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 256
#define BURST_SIZE 32

#define NUM_FRAMES 16
#define MAGIC 0x1234ABCD
#define TARGET_UDP_PORT 5000

#define MAX_FRAME_BYTES 150528
#define MAX_SEGMENTS 108

// ------------------------------------------------------------
// Custom image packet header
// Must match generate_pcap.py:
//
// magic       : 4 bytes
// frame_id    : 4 bytes
// seq         : 2 bytes
// nseg        : 2 bytes
// offset      : 4 bytes
// length      : 2 bytes
// frame_bytes : 4 bytes
//
// Total = 22 bytes
// ------------------------------------------------------------

struct __attribute__((packed)) image_header {
    uint32_t magic;
    uint32_t frame_id;
    uint16_t seq;
    uint16_t nseg;
    uint32_t offset;
    uint16_t length;
    uint32_t frame_bytes;
};


// ------------------------------------------------------------
// Ethernet / IPv4 / UDP header structures
// ------------------------------------------------------------

struct __attribute__((packed)) ethernet_header {
    uint8_t dst[6];
    uint8_t src[6];
    uint16_t ether_type;
};

struct __attribute__((packed)) ipv4_header {
    uint8_t version_ihl;
    uint8_t tos;
    uint16_t total_length;
    uint16_t identification;
    uint16_t flags_fragment;
    uint8_t ttl;
    uint8_t protocol;
    uint16_t checksum;
    uint32_t src_addr;
    uint32_t dst_addr;
};

struct __attribute__((packed)) udp_header {
    uint16_t src_port;
    uint16_t dst_port;
    uint16_t length;
    uint16_t checksum;
};


// ------------------------------------------------------------
// Frame reassembly state
// ------------------------------------------------------------

struct frame_state {
    uint8_t *data;

    uint8_t received[MAX_SEGMENTS];

    uint16_t received_count;

    uint16_t expected_segments;

    uint32_t frame_bytes;

    int initialized;
    int complete;
};


static struct frame_state frames[NUM_FRAMES];


// ------------------------------------------------------------
// Initialize frame state
// ------------------------------------------------------------

static void init_frames(void)
{
    for (int i = 0; i < NUM_FRAMES; i++) {

        frames[i].data = malloc(MAX_FRAME_BYTES);

        if (!frames[i].data) {
            fprintf(stderr, "Failed to allocate frame %d\n", i);
            exit(EXIT_FAILURE);
        }

        memset(frames[i].data, 0, MAX_FRAME_BYTES);
        memset(frames[i].received, 0, MAX_SEGMENTS);

        frames[i].received_count = 0;
        frames[i].expected_segments = 0;
        frames[i].frame_bytes = 0;

        frames[i].initialized = 0;
        frames[i].complete = 0;
    }
}


// ------------------------------------------------------------
// Free frame memory
// ------------------------------------------------------------

static void free_frames(void)
{
    for (int i = 0; i < NUM_FRAMES; i++) {
        free(frames[i].data);
    }
}


// ------------------------------------------------------------
// Write reconstructed frame to file
// ------------------------------------------------------------

static void write_frame(uint32_t frame_id)
{
    char filename[256];

    snprintf(
        filename,
        sizeof(filename),
        "reconstructed/frame_%02u.bin",
        frame_id
    );

    FILE *f = fopen(filename, "wb");

    if (!f) {
        perror("fopen");
        return;
    }

    fwrite(
        frames[frame_id].data,
        1,
        frames[frame_id].frame_bytes,
        f
    );

    fclose(f);

    printf(
        "Frame %u complete: %u bytes -> %s\n",
        frame_id,
        frames[frame_id].frame_bytes,
        filename
    );
}


// ------------------------------------------------------------
// Process one packet
// ------------------------------------------------------------

static void process_packet(
    struct rte_mbuf *mbuf,
    uint64_t *valid_packets,
    uint64_t *ignored_packets,
    uint64_t *duplicate_packets,
    uint32_t *complete_frames
)
{
    uint8_t *pkt =
        rte_pktmbuf_mtod(mbuf, uint8_t *);

    uint32_t pkt_len =
        rte_pktmbuf_pkt_len(mbuf);


    // --------------------------------------------------------
    // Ethernet header
    // --------------------------------------------------------

    if (pkt_len < sizeof(struct ethernet_header)) {
        (*ignored_packets)++;
        return;
    }

    struct ethernet_header *eth =
        (struct ethernet_header *)pkt;

    uint16_t ether_type =
        ntohs(eth->ether_type);

    // IPv4 EtherType = 0x0800
    if (ether_type != 0x0800) {
        (*ignored_packets)++;
        return;
    }


    // --------------------------------------------------------
    // IPv4 header
    // --------------------------------------------------------

    uint32_t offset =
        sizeof(struct ethernet_header);

    if (
        pkt_len <
        offset + sizeof(struct ipv4_header)
    ) {
        (*ignored_packets)++;
        return;
    }

    struct ipv4_header *ip =
        (struct ipv4_header *)(pkt + offset);

    uint8_t version =
        ip->version_ihl >> 4;

    uint8_t ihl =
        (ip->version_ihl & 0x0F) * 4;

    if (version != 4) {
        (*ignored_packets)++;
        return;
    }

    if (ihl < 20) {
        (*ignored_packets)++;
        return;
    }

    // UDP protocol number = 17
    if (ip->protocol != 17) {
        (*ignored_packets)++;
        return;
    }

    offset += ihl;


    // --------------------------------------------------------
    // UDP header
    // --------------------------------------------------------

    if (
        pkt_len <
        offset + sizeof(struct udp_header)
    ) {
        (*ignored_packets)++;
        return;
    }

    struct udp_header *udp =
        (struct udp_header *)(pkt + offset);

    uint16_t dst_port =
        ntohs(udp->dst_port);

    if (dst_port != TARGET_UDP_PORT) {
        (*ignored_packets)++;
        return;
    }

    offset += sizeof(struct udp_header);


    // --------------------------------------------------------
    // Custom image header
    // --------------------------------------------------------

    if (
        pkt_len <
        offset + sizeof(struct image_header)
    ) {
        (*ignored_packets)++;
        return;
    }

    struct image_header *hdr =
        (struct image_header *)(pkt + offset);

    uint32_t magic =
        ntohl(hdr->magic);

    uint32_t frame_id =
        ntohl(hdr->frame_id);

    uint16_t seq =
        ntohs(hdr->seq);

    uint16_t nseg =
        ntohs(hdr->nseg);

    uint32_t image_offset =
        ntohl(hdr->offset);

    uint16_t length =
        ntohs(hdr->length);

    uint32_t frame_bytes =
        ntohl(hdr->frame_bytes);


    // --------------------------------------------------------
    // Validate custom header
    // --------------------------------------------------------

    if (magic != MAGIC) {
        (*ignored_packets)++;
        return;
    }

    if (frame_id >= NUM_FRAMES) {
        (*ignored_packets)++;
        return;
    }

    if (seq >= MAX_SEGMENTS) {
        (*ignored_packets)++;
        return;
    }

    if (nseg > MAX_SEGMENTS) {
        (*ignored_packets)++;
        return;
    }

    if (frame_bytes > MAX_FRAME_BYTES) {
        (*ignored_packets)++;
        return;
    }

    if (
        image_offset + length >
        frame_bytes
    ) {
        (*ignored_packets)++;
        return;
    }


    offset += sizeof(struct image_header);


    if (pkt_len < offset + length) {
        (*ignored_packets)++;
        return;
    }


    // --------------------------------------------------------
    // Initialize frame on first packet
    // --------------------------------------------------------

    struct frame_state *frame =
        &frames[frame_id];

    if (!frame->initialized) {

        frame->initialized = 1;

        frame->expected_segments = nseg;
        frame->frame_bytes = frame_bytes;
    }


    // --------------------------------------------------------
    // Ignore duplicate chunks
    // --------------------------------------------------------

    if (frame->received[seq]) {

        (*duplicate_packets)++;

        return;
    }


    // --------------------------------------------------------
    // Copy image chunk into correct location
    // --------------------------------------------------------

    uint8_t *image_data =
        pkt + offset;

    memcpy(
        frame->data + image_offset,
        image_data,
        length
    );

    frame->received[seq] = 1;
    frame->received_count++;

    (*valid_packets)++;


    // --------------------------------------------------------
    // Check whether frame is complete
    // --------------------------------------------------------

    if (
        !frame->complete &&
        frame->received_count ==
        frame->expected_segments
    ) {
        frame->complete = 1;

        write_frame(frame_id);

        (*complete_frames)++;
    }
}


// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char **argv)
{
    int ret =
        rte_eal_init(argc, argv);

    if (ret < 0) {
        rte_exit(
            EXIT_FAILURE,
            "EAL init failed\n"
        );
    }


    // --------------------------------------------------------
    // Create mbuf pool
    // --------------------------------------------------------

    struct rte_mempool *pool =
        rte_pktmbuf_pool_create(
            "mbufs",
            NUM_MBUFS,
            MBUF_CACHE_SIZE,
            0,
            RTE_MBUF_DEFAULT_BUF_SIZE,
            rte_socket_id()
        );

    if (!pool) {
        rte_exit(
            EXIT_FAILURE,
            "Cannot create mbuf pool\n"
        );
    }


    // --------------------------------------------------------
    // Configure DPDK port
    // --------------------------------------------------------

    struct rte_eth_conf conf;
    memset(&conf, 0, sizeof(conf));

    if (
        rte_eth_dev_configure(
            0,
            1,
            1,
            &conf
        ) < 0
    ) {
        rte_exit(
            EXIT_FAILURE,
            "Cannot configure port\n"
        );
    }


    if (
        rte_eth_rx_queue_setup(
            0,
            0,
            RX_RING_SIZE,
            rte_socket_id(),
            NULL,
            pool
        ) < 0
    ) {
        rte_exit(
            EXIT_FAILURE,
            "Cannot setup RX queue\n"
        );
    }


    if (
        rte_eth_tx_queue_setup(
            0,
            0,
            RX_RING_SIZE,
            rte_socket_id(),
            NULL
        ) < 0
    ) {
        rte_exit(
            EXIT_FAILURE,
            "Cannot setup TX queue\n"
        );
    }


    if (rte_eth_dev_start(0) < 0) {
        rte_exit(
            EXIT_FAILURE,
            "Cannot start port\n"
        );
    }


    // --------------------------------------------------------
    // Prepare reconstruction
    // --------------------------------------------------------

    init_frames();

    int rc = system("mkdir -p reconstructed");
    if (rc != 0){
        fprintf(stderr, "Failed to create reconstructed directory\n");
	return EXIT_FAILURE;
    }


    uint64_t packets_received = 0;
    uint64_t valid_packets = 0;
    uint64_t ignored_packets = 0;
    uint64_t duplicate_packets = 0;

    uint32_t complete_frames = 0;


    printf("Starting packet receiver...\n");


    // --------------------------------------------------------
    // Receive loop
    // --------------------------------------------------------

    struct rte_mbuf *bufs[BURST_SIZE];

    int empty_polls = 0;

    const int MAX_EMPTY_POLLS = 100000;


    while (1) {

        uint16_t n =
            rte_eth_rx_burst(
                0,
                0,
                bufs,
                BURST_SIZE
            );

        if (n == 0) {

            empty_polls++;

            // pcap PMD does not signal EOF,
            // so stop after many empty polls
            if (
                empty_polls >
                MAX_EMPTY_POLLS
            ) {
                break;
            }

            continue;
        }

        empty_polls = 0;

        packets_received += n;


        for (uint16_t i = 0; i < n; i++) {

            process_packet(
                bufs[i],
                &valid_packets,
                &ignored_packets,
                &duplicate_packets,
                &complete_frames
            );

            rte_pktmbuf_free(bufs[i]);
        }


        // We expect exactly 16 complete frames
        if (complete_frames == NUM_FRAMES) {
            break;
        }
    }


    // --------------------------------------------------------
    // Summary
    // --------------------------------------------------------

    printf("\n");
    printf("========== Summary ==========\n");
    printf(
        "Packets received : %lu\n",
        packets_received
    );

    printf(
        "Valid packets    : %lu\n",
        valid_packets
    );

    printf(
        "Ignored packets  : %lu\n",
        ignored_packets
    );

    printf(
        "Duplicate packets: %lu\n",
        duplicate_packets
    );

    printf(
        "Complete frames  : %u / %d\n",
        complete_frames,
        NUM_FRAMES
    );


    free_frames();

    rte_eth_dev_stop(0);
    rte_eth_dev_close(0);

    return 0;
}
