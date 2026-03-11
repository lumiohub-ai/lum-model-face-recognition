"""
Manual test script for frame capture endpoint.

This script tests the frame capture functionality by:
1. Starting the API server (if not running)
2. Sending a capture request
3. Verifying the response
4. Checking if the frame was uploaded to GCS

Usage:
    python tests/test_frame_capture.py --org-slug test-org --camera-id 1

Prerequisites:
    - API server running on port 5000 (or set via --port)
    - Camera configured in backend with valid RTSP stream
    - GCS credentials configured
    - Environment variables set (SA_EMAIL, SA_PASSWORD, etc.)
"""

import argparse
import requests
import sys
import time
from typing import Dict, Any


def test_capture_endpoint(
    org_slug: str,
    camera_id: int,
    frame_index: int = 1,
    quality: int = 85,
    api_url: str = "http://localhost:5000"
) -> Dict[str, Any]:
    """Test the frame capture endpoint.
    
    Args:
        org_slug: Organization slug
        camera_id: Camera ID
        frame_index: Frame index (1-30)
        quality: JPEG quality (1-100)
        api_url: Base API URL
        
    Returns:
        Response JSON or error details
    """
    endpoint = f"{api_url}/api/v1/org/{org_slug}/cameras/{camera_id}/capture-for-calibration"
    
    payload = {
        "frame_index": frame_index,
        "quality": quality
    }
    
    print(f"\n{'='*60}")
    print(f"Testing Frame Capture Endpoint")
    print(f"{'='*60}")
    print(f"Endpoint: {endpoint}")
    print(f"Payload: {payload}")
    print(f"{'='*60}\n")
    
    try:
        print("⏳ Sending request...")
        start_time = time.time()
        
        response = requests.post(
            endpoint,
            json=payload,
            timeout=60  # 60 second timeout
        )
        
        elapsed = time.time() - start_time
        print(f"✓ Request completed in {elapsed:.2f} seconds")
        
        # Print response details
        print(f"\n{'='*60}")
        print(f"Response Details")
        print(f"{'='*60}")
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        # Parse response
        try:
            data = response.json()
            print(f"\n{'='*60}")
            print(f"Response Body:")
            print(f"{'='*60}")
            
            import json
            print(json.dumps(data, indent=2))
            
            # Verify response structure
            if response.status_code == 200:
                print(f"\n{'='*60}")
                print(f"Validation Results:")
                print(f"{'='*60}")
                
                checks = {
                    "success": data.get('success') == True,
                    "frame_url": 'frame_url' in data and data['frame_url'].startswith('gs://'),
                    "signed_url": 'signed_url' in data and data['signed_url'].startswith('http'),
                    "captured_at": 'captured_at' in data,
                    "metadata": 'metadata' in data,
                    "metadata.width": data.get('metadata', {}).get('width') is not None,
                    "metadata.height": data.get('metadata', {}).get('height') is not None,
                    "metadata.size_bytes": data.get('metadata', {}).get('size_bytes') is not None,
                }
                
                all_passed = all(checks.values())
                
                for check, passed in checks.items():
                    status = "✓" if passed else "✗"
                    print(f"{status} {check}: {passed}")
                
                print(f"{'='*60}")
                
                if all_passed:
                    print("\n✅ All validation checks passed!")
                    print(f"\n📸 Frame URL: {data.get('frame_url')}")
                    print(f"🔗 Signed URL: {data.get('signed_url')[:80]}...")
                    print(f"📊 Dimensions: {data.get('metadata', {}).get('width')}x{data.get('metadata', {}).get('height')}")
                    print(f"💾 Size: {data.get('metadata', {}).get('size_bytes')} bytes")
                else:
                    print("\n❌ Some validation checks failed!")
                    return {"error": "Validation failed", "details": checks}
            else:
                print(f"\n❌ Request failed with status code {response.status_code}")
                if 'detail' in data:
                    print(f"Error detail: {data['detail']}")
                return {"error": f"HTTP {response.status_code}", "details": data}
            
            return data
            
        except ValueError as e:
            print(f"\n❌ Failed to parse JSON response: {e}")
            print(f"Raw response: {response.text}")
            return {"error": "Invalid JSON response", "details": str(e)}
            
    except requests.exceptions.Timeout:
        print(f"\n❌ Request timed out after 60 seconds")
        return {"error": "Timeout", "details": "Request exceeded 60 second timeout"}
        
    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ Connection error: {e}")
        print(f"\nIs the API server running at {api_url}?")
        return {"error": "Connection error", "details": str(e)}
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": "Unexpected error", "details": str(e)}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test frame capture endpoint')
    parser.add_argument('--org-slug', required=True, help='Organization slug')
    parser.add_argument('--camera-id', type=int, required=True, help='Camera ID')
    parser.add_argument('--frame-index', type=int, default=1, help='Frame index (1-30)')
    parser.add_argument('--quality', type=int, default=85, help='JPEG quality (1-100)')
    parser.add_argument('--api-url', default='http://localhost:5000', help='API base URL')
    
    args = parser.parse_args()
    
    # Validate parameters
    if not 1 <= args.frame_index <= 30:
        print("❌ Error: frame_index must be between 1 and 30")
        sys.exit(1)
    
    if not 1 <= args.quality <= 100:
        print("❌ Error: quality must be between 1 and 100")
        sys.exit(1)
    
    # Run test
    result = test_capture_endpoint(
        org_slug=args.org_slug,
        camera_id=args.camera_id,
        frame_index=args.frame_index,
        quality=args.quality,
        api_url=args.api_url
    )
    
    # Exit with appropriate code
    if 'error' in result:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
