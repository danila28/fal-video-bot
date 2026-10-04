#!/usr/bin/env python3
"""Smoke test: verify LLM service routing and provider basic functionality.

Run with:
  OPENAI_API_KEY=sk-proj-... python scripts/smoke_llm.py
Or with a photo:
  OPENAI_API_KEY=sk-proj-... python scripts/smoke_llm.py --photo /path/to/photo.jpg

This script tests:
1. Basic routing (model selection, default model)
2. Text generation with Luna, Terra, Sol models
3. Vision capability with local image (if --photo provided)
4. Error handling for refusals and empty responses
"""

import asyncio
import os
import sys
import argparse
import logging

# Add parent directory to path so we can import services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.llm import LLMService, OpenAITextProvider, GeminiTextProvider
from services.llm import LLMEmptyResponse, LLMRefusal, LLMError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def smoke_test(photo_path: str = None):
    """Run smoke tests on LLM service."""

    # Initialize providers
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not openai_key and not gemini_key:
        print("❌ At least one of OPENAI_API_KEY or GEMINI_API_KEY must be set")
        return False

    providers = []
    if openai_key:
        providers.append(OpenAITextProvider(api_key=openai_key))
        print("✅ OpenAI provider loaded")
    if gemini_key:
        providers.append(GeminiTextProvider(api_key=gemini_key))
        print("✅ Gemini provider loaded")

    # Create LLM service
    llm = LLMService(providers=providers, default_model="terra")
    print(f"✅ LLM service initialized (default: terra)")

    # Test 1: List available models
    models = llm.get_text_models()
    print(f"\n📋 Available models ({len(models)}):")
    for m in models:
        print(f"  • {m['name']:<30} {m['price']}")

    # Test 2: Basic text generation with Terra
    if "terra" in [m["name"] for m in models]:
        print("\n🔄 Testing text generation (terra)...")
        try:
            response = await llm.generate_text(
                prompt="Write a haiku about cats.",
                model="terra",
            )
            if response:
                print(f"✅ Text generation successful:\n{response[:100]}...")
            else:
                print("⚠️  Empty response")
        except (LLMEmptyResponse, LLMRefusal) as e:
            print(f"⚠️  Model limitation: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")

    # Test 3: Vision test with image (if provided)
    if photo_path:
        if not os.path.exists(photo_path):
            print(f"\n❌ Photo not found: {photo_path}")
            return False

        # Find first vision-capable model
        vision_models = [m for m in models if m.get("vision")]
        if not vision_models:
            print("\n⚠️  No vision-capable models available")
        else:
            vision_model = vision_models[0]["name"]
            print(f"\n🖼️  Testing vision capability ({vision_model})...")
            try:
                response = await llm.generate_text(
                    prompt="Describe this image briefly.",
                    model=vision_model,
                    image_paths=[photo_path],
                )
                if response:
                    print(f"✅ Vision test successful:\n{response[:100]}...")
                else:
                    print("⚠️  Empty response")
            except LLMRefusal as e:
                print(f"⚠️  Model refused (common for real people in photos): {e}")
            except (LLMEmptyResponse, LLMError) as e:
                print(f"⚠️  Model limitation: {e}")
            except Exception as e:
                print(f"❌ Error: {e}")

    # Test 4: Unknown model fallback
    print("\n🔄 Testing unknown model fallback...")
    try:
        response = await llm.generate_text(
            prompt="Hi",
            model="unknown-model-xyz",
        )
        print(f"✅ Fallback to default successful")
    except Exception as e:
        print(f"❌ Fallback failed: {e}")

    print("\n✅ Smoke test completed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for LLM service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  OPENAI_API_KEY=sk-proj-... python scripts/smoke_llm.py
  OPENAI_API_KEY=sk-proj-... python scripts/smoke_llm.py --photo ~/my_photo.jpg
        """
    )
    parser.add_argument("--photo", help="Path to image file for vision test")
    args = parser.parse_args()

    success = asyncio.run(smoke_test(photo_path=args.photo))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
