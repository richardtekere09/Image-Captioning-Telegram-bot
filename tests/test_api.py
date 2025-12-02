"""
API Test Script
===============

Tests the FastAPI service with real images.

Usage:
    1. Start server: python fastapi_service.py
    2. Run tests: python test_api.py
"""

import base64
import requests
import json
import time
from pathlib import Path


# Configuration
API_BASE_URL = "http://localhost:8000"
CAT_IMAGE_PATH = "data/sample_images/cat_demo.jpeg"


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def load_image_as_base64(image_path):
    """Load image and convert to base64."""
    with open(image_path, "rb") as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode("utf-8")


def test_health_check():
    """Test health check endpoint."""
    print_section("TEST 1: Health Check")

    try:
        response = requests.get(f"{API_BASE_URL}/health")

        print(f"Status Code: {response.status_code}")
        print(f"Response:")
        print(json.dumps(response.json(), indent=2))

        if response.status_code == 200:
            print("✅ Health check passed!")
            return True
        else:
            print("❌ Health check failed!")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API!")
        print("   Make sure server is running: python fastapi_service.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_single_inference():
    """Test single image caption generation."""
    print_section("TEST 2: Single Image Inference")

    try:
        # Load image
        print(f"📸 Loading image: {CAT_IMAGE_PATH}")
        image_base64 = load_image_as_base64(CAT_IMAGE_PATH)
        print(f"✅ Image loaded: {len(image_base64)} characters (base64)")

        # Prepare request
        request_data = {"image": image_base64, "max_length": 50, "num_beams": 3}

        # Send request
        print("\n🚀 Sending request to API...")
        start_time = time.time()

        response = requests.post(
            f"{API_BASE_URL}/api/v1/infer",
            json=request_data,
            timeout=60,  # Wait up to 60 seconds
        )

        request_time = time.time() - start_time

        # Display results
        print(f"\n📊 Results:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Request Time: {request_time:.2f}s (includes network)")

        if response.status_code == 200:
            result = response.json()
            print(f"\n   📝 Caption: '{result['caption']}'")
            print(f"   ⏱️  Processing Time: {result['processing_time']:.2f}s")
            print(f"   🧠 Model: {result['model_name']}")
            print(f"   💻 Device: {result['device']}")
            print("\n✅ Single inference test passed!")
            return True
        else:
            print(f"\n❌ Request failed!")
            print(f"   Error: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Request timed out! (Model might be loading...)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_different_parameters():
    """Test with different parameters."""
    print_section("TEST 3: Different Parameters")

    try:
        image_base64 = load_image_as_base64(CAT_IMAGE_PATH)

        # Test different num_beams
        print("\n🔬 Testing different num_beams values...\n")

        for num_beams in [1, 3, 5]:
            print(f"   Testing num_beams={num_beams}...")

            response = requests.post(
                f"{API_BASE_URL}/api/v1/infer",
                json={"image": image_base64, "max_length": 50, "num_beams": num_beams},
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Caption: '{result['caption']}'")
                print(f"      Time: {result['processing_time']:.2f}s\n")
            else:
                print(f"   ❌ Failed: {response.status_code}\n")

        print("✅ Parameter test completed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_batch_inference():
    """Test batch processing with multiple images."""
    print_section("TEST 4: Batch Inference")

    try:
        # Load same image multiple times (simulating multiple images)
        image_base64 = load_image_as_base64(CAT_IMAGE_PATH)

        print("📦 Preparing batch request (3 images)...")

        request_data = {
            "images": [image_base64, image_base64, image_base64],
            "max_length": 50,
            "num_beams": 3,
        }

        print("🚀 Sending batch request...")
        start_time = time.time()

        response = requests.post(
            f"{API_BASE_URL}/api/v1/infer/batch",
            json=request_data,
            timeout=120,  # Longer timeout for batch
        )

        request_time = time.time() - start_time

        print(f"\n📊 Results:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Request Time: {request_time:.2f}s")

        if response.status_code == 200:
            result = response.json()
            print(f"\n   Images Processed: {result['images_processed']}")
            print(f"   Total Processing Time: {result['total_processing_time']:.2f}s")
            print(
                f"   Average per Image: {result['total_processing_time'] / result['images_processed']:.2f}s"
            )

            print(f"\n   📝 Captions:")
            for idx, caption_result in enumerate(result["results"], 1):
                print(f"      {idx}. '{caption_result['caption']}'")

            print("\n✅ Batch inference test passed!")
            return True
        else:
            print(f"\n❌ Batch request failed!")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_error_handling():
    """Test API error handling."""
    print_section("TEST 5: Error Handling")

    tests_passed = 0
    total_tests = 3

    # Test 1: Invalid base64
    print("\n🧪 Test 5.1: Invalid base64 encoding...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/infer",
            json={"image": "not-valid-base64!@#$", "max_length": 50, "num_beams": 3},
        )

        if response.status_code == 422:  # Validation error
            print("   ✅ Correctly rejected invalid base64")
            tests_passed += 1
        else:
            print(f"   ❌ Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 2: Missing required field
    print("\n🧪 Test 5.2: Missing required field...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/infer",
            json={
                "max_length": 50,
                # Missing 'image' field
            },
        )

        if response.status_code == 422:
            print("   ✅ Correctly rejected missing field")
            tests_passed += 1
        else:
            print(f"   ❌ Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 3: Invalid parameter value
    print("\n🧪 Test 5.3: Invalid parameter value...")
    try:
        image_base64 = load_image_as_base64(CAT_IMAGE_PATH)
        response = requests.post(
            f"{API_BASE_URL}/api/v1/infer",
            json={
                "image": image_base64,
                "max_length": 5,  # Too low (min is 10)
                "num_beams": 3,
            },
        )

        if response.status_code == 422:
            print("   ✅ Correctly rejected invalid parameter")
            tests_passed += 1
        else:
            print(f"   ❌ Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print(f"\n   Error Handling: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests


def test_api_documentation():
    """Check if API documentation is available."""
    print_section("TEST 6: API Documentation")

    try:
        # Swagger UI
        response = requests.get(f"{API_BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ Swagger UI available at: http://localhost:8000/docs")
        else:
            print("❌ Swagger UI not available")

        # ReDoc
        response = requests.get(f"{API_BASE_URL}/redoc")
        if response.status_code == 200:
            print("✅ ReDoc available at: http://localhost:8000/redoc")
        else:
            print("❌ ReDoc not available")

        # OpenAPI schema
        response = requests.get(f"{API_BASE_URL}/openapi.json")
        if response.status_code == 200:
            print("✅ OpenAPI schema available at: http://localhost:8000/openapi.json")
        else:
            print("❌ OpenAPI schema not available")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  🚀 FastAPI Service Test Suite")
    print("=" * 60)
    print(f"\n📍 API URL: {API_BASE_URL}")
    print(f"📸 Test Image: {CAT_IMAGE_PATH}")

    # Check if image exists
    if not Path(CAT_IMAGE_PATH).exists():
        print(f"\n❌ Error: Image not found at {CAT_IMAGE_PATH}")
        return

    # Run tests
    results = []

    results.append(("Health Check", test_health_check()))

    if results[0][1]:  # Only continue if health check passed
        results.append(("Single Inference", test_single_inference()))
        results.append(("Different Parameters", test_different_parameters()))
        results.append(("Batch Inference", test_batch_inference()))
        results.append(("Error Handling", test_error_handling()))
        results.append(("API Documentation", test_api_documentation()))

    # Summary
    print_section("TEST SUMMARY")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print("\n📊 Results:")
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")

    print(f"\n   Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! API is working correctly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
