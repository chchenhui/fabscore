#!/usr/bin/env python3
"""
Download and merge SAE feature data from Neuronpedia S3 bucket.

This script downloads all feature data from the Neuronpedia S3 bucket for
Gemma-2-9B layer 32 SAE features and merges them into a single JSONL file.

The data includes feature information like descriptions, activation patterns,
and other metadata for all 131,072 features.

Usage:
    python taboo/download_sae_features.py --output_file merged_features.jsonl
"""
import argparse
import gzip
import json
import os
import sys
import xml.etree.ElementTree as ET
from typing import List
from urllib.parse import urljoin

import requests

# Try to import tqdm, use simple fallback if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="Processing", **kwargs):
        total = len(iterable) if hasattr(iterable, '__len__') else None
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc}: {i+1}/{total}", end="", flush=True)
            else:
                print(f"\r{desc}: {i+1}", end="", flush=True)
            yield item
        print()  # New line at the end


def list_s3_bucket_contents(bucket_name: str, prefix: str = "", max_keys: int = 1000) -> List[str]:
    """List contents of S3 bucket using the S3 API directly."""
    try:
        # Use S3 list-objects API
        base_url = f"https://{bucket_name}.s3.amazonaws.com"
        params = {
            'list-type': '2',
            'prefix': prefix,
            'max-keys': max_keys
        }

        all_keys = []
        continuation_token = None

        while True:
            if continuation_token:
                params['continuation-token'] = continuation_token

            response = requests.get(base_url, params=params)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.content)

            # Extract namespace
            namespace = root.tag.split('}')[0] + '}' if '}' in root.tag else ''

            # Find all Contents elements (files)
            for content in root.findall(f'{namespace}Contents'):
                key_elem = content.find(f'{namespace}Key')
                if key_elem is not None:
                    all_keys.append(key_elem.text)

            # Check if there are more results
            is_truncated = root.find(f'{namespace}IsTruncated')
            if is_truncated is None or is_truncated.text.lower() != 'true':
                break

            # Get continuation token for next page
            next_token = root.find(f'{namespace}NextContinuationToken')
            if next_token is not None:
                continuation_token = next_token.text
            else:
                break

        return all_keys

    except requests.RequestException as e:
        print(f"Error fetching bucket contents: {e}")
        return []
    except ET.ParseError as e:
        print(f"Error parsing S3 response: {e}")
        return []


def download_file(bucket_name: str, key: str, local_path: str) -> bool:
    """Download a file from S3 bucket to local path."""
    try:
        url = f"https://{bucket_name}.s3.amazonaws.com/{key}"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download with simple progress reporting
        total_size = int(response.headers.get('content-length', 0))

        with open(local_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                filename = os.path.basename(local_path)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r  {filename}: {percent:.1f}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)", end="", flush=True)
                print()  # New line when done

        return True

    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False


def merge_jsonl_files(input_files: List[str], output_file: str) -> None:
    """Merge multiple JSONL files (regular or gzipped) into a single JSONL file."""
    print(f"Merging {len(input_files)} files into {output_file}")

    total_lines = 0

    with open(output_file, 'w', encoding='utf-8') as outf:
        for input_file in tqdm(input_files, desc="Merging files"):
            if not os.path.exists(input_file):
                print(f"Warning: File {input_file} not found, skipping")
                continue

            try:
                # Check if file is gzipped
                if input_file.endswith('.gz'):
                    file_handle = gzip.open(input_file, 'rt', encoding='utf-8')
                else:
                    file_handle = open(input_file, 'r', encoding='utf-8')

                with file_handle as inf:
                    for line in inf:
                        line = line.strip()
                        if line:  # Skip empty lines
                            # Validate JSON
                            try:
                                json.loads(line)
                                outf.write(line + '\n')
                                total_lines += 1
                            except json.JSONDecodeError:
                                print(f"Warning: Invalid JSON line in {input_file}: {line[:100]}...")
                                continue
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                continue

    print(f"Merged {total_lines} feature records into {output_file}")


def main():
    """Main function to download and merge SAE feature data."""
    parser = argparse.ArgumentParser(
        description="Download and merge SAE feature data from Neuronpedia S3 bucket."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="merged_sae_features.jsonl",
        help="Output JSONL file path (default: merged_sae_features.jsonl)."
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default="temp_sae_downloads",
        help="Temporary directory for downloads (default: temp_sae_downloads)."
    )
    parser.add_argument(
        "--keep_downloads",
        action="store_true",
        help="Keep downloaded files after merging (default: delete)."
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=32,
        help="SAE layer number (default: 32)."
    )
    parser.add_argument(
        "--width_k",
        type=int,
        default=131,
        help="SAE width in thousands (default: 131)."
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to download (for testing)."
    )

    args = parser.parse_args()

    # S3 bucket configuration
    bucket_name = "neuronpedia-datasets"
    sae_prefix = f"v1/gemma-2-9b/{args.layer}-gemmascope-res-{args.width_k}k/features/"

    print(f"Downloading SAE feature data for Gemma-2-9B layer {args.layer}")
    print(f"SAE width: {args.width_k}k features")
    print(f"S3 bucket: {bucket_name}")
    print(f"Prefix: {sae_prefix}")
    print()

    # List all files in the features subdirectory
    print("Listing files in features subdirectory...")
    all_files = list_s3_bucket_contents(bucket_name, sae_prefix, max_keys=10000)

    if not all_files:
        print("Error: No files found in features subdirectory")
        sys.exit(1)

    print(f"Found {len(all_files)} files in features subdirectory")

    # Filter for JSONL files (regular and gzipped) and JSON files
    data_files = [f for f in all_files if f.endswith('.jsonl') or f.endswith('.json') or f.endswith('.jsonl.gz')]

    if not data_files:
        print("No JSONL/JSON files found in features subdirectory")
        # Show what files we did find
        print("Found files:")
        for f in all_files[:10]:  # Show first 10
            print(f"  {f}")
        if len(all_files) > 10:
            print(f"  ... and {len(all_files) - 10} more")
        sys.exit(1)

    print(f"Found {len(data_files)} data files to download")

    # Limit files for testing if requested
    if args.max_files and args.max_files < len(data_files):
        data_files = data_files[:args.max_files]
        print(f"Limited to first {args.max_files} files for testing")

    # Show some example files
    print("Example files:")
    for f in data_files[:5]:
        print(f"  {f}")
    if len(data_files) > 5:
        print(f"  ... and {len(data_files) - 5} more")

    # Create download directory
    os.makedirs(args.download_dir, exist_ok=True)

    # Download all data files
    downloaded_files = []

    print("Downloading feature files...")
    for file_key in tqdm(data_files, desc="Downloading files"):
        # Local file path
        filename = os.path.basename(file_key)
        local_file = os.path.join(args.download_dir, filename)

        # Download file
        if download_file(bucket_name, file_key, local_file):
            downloaded_files.append(local_file)
        else:
            print(f"Failed to download {file_key}")

    if not downloaded_files:
        print("Error: No files were successfully downloaded")
        sys.exit(1)

    print(f"Successfully downloaded {len(downloaded_files)} files")

    # Merge all downloaded data files
    print("Merging downloaded files...")
    merge_jsonl_files(downloaded_files, args.output_file)

    # Clean up downloaded files unless requested to keep them
    if not args.keep_downloads:
        print("Cleaning up temporary files...")
        for file_path in downloaded_files:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Warning: Could not remove {file_path}: {e}")

        # Try to remove the download directory if it's empty
        try:
            os.rmdir(args.download_dir)
        except OSError:
            pass  # Directory not empty or other issue

    print(f"Feature data merged successfully into {args.output_file}")

    # Show file size and line count
    if os.path.exists(args.output_file):
        file_size = os.path.getsize(args.output_file)
        print(f"Output file size: {file_size / (1024*1024):.1f} MB")

        # Count lines
        try:
            with open(args.output_file, 'r') as f:
                line_count = sum(1 for line in f if line.strip())
            print(f"Total feature records: {line_count:,}")
        except Exception as e:
            print(f"Could not count lines: {e}")


if __name__ == "__main__":
    main()
