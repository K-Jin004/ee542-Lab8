import os
import math
import random
import struct
import numpy as np

from scapy.all import Ether, IP, UDP, Raw, wrpcap


# ============================================================
# Configuration
# ============================================================

NUM_IMAGES = 16
HEIGHT = 224
WIDTH = 224
CHANNELS = 3

CHUNK_SIZE = 1400

ORIGINAL_DIR = "original_images"

# Custom packet identifier
MAGIC = 0x1234ABCD

SRC_MAC = "02:00:00:00:00:01"
DST_MAC = "02:00:00:00:00:02"

SRC_IP = "10.0.0.1"
DST_IP = "10.0.0.2"

SRC_PORT = 40000
DST_PORT = 5000


# ============================================================
# Custom header format
#
# magic       : 4 bytes
# frame_id    : 4 bytes
# seq         : 2 bytes
# nseg        : 2 bytes
# offset      : 4 bytes
# length      : 2 bytes
# frame_bytes : 4 bytes
#
# Total = 22 bytes
#
# "!" means network byte order (big-endian)
# I = unsigned int    = 4 bytes
# H = unsigned short  = 2 bytes
# ============================================================

HEADER_FORMAT = "!IIHHIHI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


# ============================================================
# Generate test images
# ============================================================

def generate_test_images():
    """
    Generate 16 random 224x224 RGB images.

    Each image is stored as raw binary so that Part 4 can later
    compare the reconstructed image byte-for-byte with the original.
    """

    os.makedirs(ORIGINAL_DIR, exist_ok=True)

    images = []

    for frame_id in range(NUM_IMAGES):

        # Random RGB image:
        # shape = (224, 224, 3)
        # each channel is uint8: 0 to 255
        image = np.random.randint(
            0,
            256,
            size=(HEIGHT, WIDTH, CHANNELS),
            dtype=np.uint8
        )

        # Convert NumPy image array into raw bytes
        image_bytes = image.tobytes()

        # Save original image bytes
        path = os.path.join(
            ORIGINAL_DIR,
            f"frame_{frame_id:02d}.bin"
        )

        with open(path, "wb") as f:
            f.write(image_bytes)

        images.append(image_bytes)

        print(
            f"Generated frame {frame_id}: "
            f"{len(image_bytes)} bytes"
        )

    return images


# ============================================================
# Build UDP packets
# ============================================================

def build_packets(images):
    """
    Split each image into 1400-byte chunks.

    Each chunk becomes one UDP packet.

    The UDP payload contains:

        custom header + image chunk
    """

    packets = []

    for frame_id, image_bytes in enumerate(images):

        frame_bytes = len(image_bytes)

        # Number of packets required for this image
        nseg = math.ceil(frame_bytes / CHUNK_SIZE)

        for seq in range(nseg):

            # Where this chunk belongs in the original image
            offset = seq * CHUNK_SIZE

            # Extract at most 1400 image bytes
            chunk = image_bytes[
                offset : offset + CHUNK_SIZE
            ]

            length = len(chunk)

            # ------------------------------------------------
            # Build custom binary header
            # ------------------------------------------------

            header = struct.pack(
                HEADER_FORMAT,
                MAGIC,
                frame_id,
                seq,
                nseg,
                offset,
                length,
                frame_bytes
            )

            # UDP payload:
            #
            # [ custom header ][ image bytes ]
            payload = header + chunk

            # ------------------------------------------------
            # Build Ethernet / IPv4 / UDP packet
            # ------------------------------------------------

            packet = (
                Ether(
                    src=SRC_MAC,
                    dst=DST_MAC
                )
                /
                IP(
                    src=SRC_IP,
                    dst=DST_IP
                )
                /
                UDP(
                    sport=SRC_PORT,
                    dport=DST_PORT
                )
                /
                Raw(payload)
            )

            packets.append(packet)

        print(
            f"Frame {frame_id}: "
            f"{nseg} packets"
        )

    return packets


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Step 1: Generate ONE set of 16 images
    # --------------------------------------------------------

    print("Generating test images...")

    images = generate_test_images()

    # --------------------------------------------------------
    # Step 2: Convert those SAME images into UDP packets
    # --------------------------------------------------------

    print()
    print("Building UDP packets...")

    packets = build_packets(images)

    print()
    print(f"Total packets: {len(packets)}")

    # --------------------------------------------------------
    # Step 3: Save ordered PCAP
    # --------------------------------------------------------

    wrpcap(
        "images_ordered.pcap",
        packets
    )

    print("Saved images_ordered.pcap")

    # --------------------------------------------------------
    # Step 4: Make a COPY of the exact same packets,
    #         shuffle only their order,
    #         then save another PCAP
    # --------------------------------------------------------

    shuffled_packets = packets.copy()

    random.shuffle(shuffled_packets)

    wrpcap(
        "images_shuffled.pcap",
        shuffled_packets
    )

    print("Saved images_shuffled.pcap")

    # --------------------------------------------------------
    # Step 5: Print header and MTU information
    # --------------------------------------------------------

    print()
    print(
        f"Custom header size: "
        f"{HEADER_SIZE} bytes"
    )

    ethernet_header = 14
    ipv4_header = 20
    udp_header = 8

    max_payload = (
        HEADER_SIZE
        + CHUNK_SIZE
    )

    total_frame_size = (
        ethernet_header
        + ipv4_header
        + udp_header
        + max_payload
    )

    print(
        f"Maximum Ethernet frame size: "
        f"{total_frame_size} bytes"
    )

    print()

    if total_frame_size <= 1500:
        print("Packet fits within 1500-byte MTU.")
    else:
        print("WARNING: Packet exceeds 1500-byte MTU.")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("Done.")
    print(
        f"Original images: "
        f"{ORIGINAL_DIR}/"
    )
    print(
        "Ordered PCAP: "
        "images_ordered.pcap"
    )
    print(
        "Shuffled PCAP: "
        "images_shuffled.pcap"
    )


if __name__ == "__main__":
    main()
