"""
Generate synthetic safety training data for nanochat.

This script generates diverse safety conversations covering:
- Harmful request refusals (violence, illegal activities, hate speech, etc.)
- Safe request handling
- Edge cases and legitimate use scenarios

Design features:
- Multi-category coverage across safety dimensions
- Balanced sampling across categories and severities
- Quality filtering with length and validation checks
- Rich metadata for analysis and tracking

Usage:
    python dev/gen_safety_data.py [--num_conversations 500] [--num_workers 4]

Requirements:
    - OpenRouter API key in 'openroutertoken.txt' (same as gen_synthetic_data.py)
    - Or set OPENROUTER_API_KEY environment variable
"""

import os
import sys
import json
import random
import requests
import argparse
import time
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add nanochat to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.common import get_base_dir

# Import our config and templates
import safety_config as config
import safety_templates as templates


# ============================================================================
# API Setup
# ============================================================================

def get_api_key():
    """Get OpenRouter API key from file or environment."""
    # Try file first (same as gen_synthetic_data.py)
    token_file = Path(__file__).parent.parent / "openroutertoken.txt"
    if token_file.exists():
        return token_file.read_text().strip()
    
    # Try environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key
    
    raise FileNotFoundError(
        "OpenRouter API key not found. Either:\n"
        "1. Create 'openroutertoken.txt' in repo root, OR\n"
        "2. Set OPENROUTER_API_KEY environment variable"
    )


API_KEY = get_api_key()
API_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


# ============================================================================
# Data Generation
# ============================================================================

def generate_single_conversation(idx, category, scenario_type, severity, max_retries=3):
    """
    Generate a single safety conversation with auto-retry on failure.
    
    Args:
        idx: Index for random seeding
        category: Safety category
        scenario_type: Scenario type
        severity: Severity level
        max_retries: Maximum retry attempts (default: 3)
        
    Returns:
        (conversation_dict, error_message) tuple
    """
    
    # Use idx as seed for reproducible diversity
    rng = random.Random(idx)
    diversity_seeds = rng.sample(config.USER_OPENING_STYLES, k=5)
    
    # Create generation prompt
    prompt = templates.format_generation_prompt(
        category, scenario_type, severity, diversity_seeds
    )
    
    # Retry loop
    for attempt in range(max_retries):
        try:
            # Call API
            payload = {
                "model": config.API_CONFIG["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.API_CONFIG["temperature"],
                "max_tokens": config.API_CONFIG["max_tokens"],
                "response_format": {"type": "json_object"},
            }
            
            response = requests.post(
                config.API_CONFIG["url"],
                headers=API_HEADERS,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # Parse response
            content = result["choices"][0]["message"]["content"]
            
            # Clean up markdown code blocks
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            conversation = json.loads(content)
            
            # Validate format
            is_valid, error = templates.validate_conversation_format(conversation)
            if not is_valid:
                raise ValueError(f"Validation failed: {error}")
            
            # Add metadata and return success
            enhanced = templates.format_metadata(
                category, scenario_type, severity, conversation
            )
            
            return enhanced, None
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Simple 1 second delay
                continue
            return None, str(e)
    
    return None, "Max retries exceeded"


def generate_conversation_batch(specs):
    """
    Generate a batch of conversations based on specifications.
    
    Args:
        specs: List of (idx, category, scenario_type, severity) tuples
        
    Returns:
        List of generated conversations
    """
    
    conversations = []
    errors = defaultdict(int)
    
    print(f"Generating {len(specs)} conversations with {config.API_CONFIG['num_workers']} workers...")
    
    with ThreadPoolExecutor(max_workers=config.API_CONFIG["num_workers"]) as executor:
        # Submit all tasks
        futures = [
            executor.submit(generate_single_conversation, idx, cat, scen, sev)
            for idx, cat, scen, sev in specs
        ]
        
        # Process results as they complete (like gen_synthetic_data.py line 365)
        for future in as_completed(futures):
            try:
                conversation, error = future.result()
                    
                if conversation:
                    conversations.append(conversation)
                    print(f"✓ Generated {len(conversations)}/{len(specs)}")
                else:
                    errors[error] += 1
                    print(f"✗ Error: {error}")
                        
            except Exception as e:
                errors[str(e)] += 1
                print(f"✗ Exception: {e}")
    
    # Print error summary
    if errors:
        print(f"\n⚠️  Encountered {sum(errors.values())} errors:")
        for error, count in errors.items():
            print(f"  - {error}: {count}")
    
    return conversations


# ============================================================================
# Sampling Strategy
# ============================================================================

def create_generation_plan(target_num):
    """
    Create a generation plan ensuring diverse coverage.
    
    This implements a multi-dimensional sampling strategy for balancing
    multiple categories, scenarios, and severity levels.
    
    Returns:
        List of (idx, category, scenario_type, severity) specifications
    """
    
    specs = []
    idx = 0
    
    categories = list(config.SAFETY_CATEGORIES.keys())
    samples_per_category = target_num // len(categories)
    remainder = target_num % len(categories)
    
    # 准备加权采样
    scenario_types = list(config.TARGET_DISTRIBUTION["scenario_distribution"].keys())
    scenario_weights = list(config.TARGET_DISTRIBUTION["scenario_distribution"].values())
    
    severity_levels = list(config.TARGET_DISTRIBUTION["severity_distribution"].keys())
    severity_weights = list(config.TARGET_DISTRIBUTION["severity_distribution"].values())
    
    for cat_idx, category in enumerate(categories):
        num_samples = samples_per_category + (1 if cat_idx < remainder else 0)
        
        for i in range(num_samples):
            scenario = random.choices(scenario_types, weights=scenario_weights, k=1)[0]
            severity = random.choices(severity_levels, weights=severity_weights, k=1)[0]
            
            if scenario == "benign":
                severity = "benign"
            elif scenario == "jailbreak_attempt" and severity == "benign":
                severity = "high"
            
            specs.append((idx, category, scenario, severity))
            idx += 1
    
    random.Random(config.RNG_SEED).shuffle(specs)
    return specs


# ============================================================================
# Quality Control
# ============================================================================

def quality_filter_conversations(conversations, tokenizer=None):
    """
    Apply quality filters to generated conversations.
    
    Filters based on length, format validation, and content quality.
    
    Args:
        conversations: List of generated conversations
        tokenizer: Optional tokenizer for length checks
        
    Returns:
        (filtered_conversations, stats_dict)
    """
    
    # Use tiktoken for lightweight token counting (GPT-style tokenizer)
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
    except ImportError:
        # Fallback: use character-based estimation if tiktoken not available
        tokenizer = None
    
    filtered = []
    stats = defaultdict(int)
    
    for conv in conversations:
        stats["total"] += 1
        
        # Basic format validation
        is_valid, error = templates.validate_conversation_format(conv)
        if not is_valid:
            stats[f"reject_format: {error}"] += 1
            continue
        
        messages = conv["messages"]
        
        # Length checks
        user_msg = messages[0]["content"]
        asst_msg = messages[1]["content"]
        
        if len(user_msg) < config.QUALITY_THRESHOLDS["min_user_length"]:
            stats["reject_user_too_short"] += 1
            continue
        
        if len(asst_msg) < config.QUALITY_THRESHOLDS["min_assistant_length"]:
            stats["reject_assistant_too_short"] += 1
            continue
        
        # Token length check
        # Format as training example to check total length
        formatted = format_for_training(conv, tokenizer)
        num_tokens = len(tokenizer.encode(formatted))
        
        if num_tokens > config.QUALITY_THRESHOLDS["max_length_tokens"]:
            stats["reject_too_long"] += 1
            continue
        
        # Check for template-like content (simple heuristic)
        if is_too_generic(asst_msg):
            stats["reject_too_generic"] += 1
            continue
        
        # Passed all checks
        stats["accepted"] += 1
        filtered.append(conv)
    
    return filtered, dict(stats)


def format_for_training(conversation, tokenizer):
    """Format conversation as it would appear during training."""
    messages = conversation["messages"]
    # Simple concatenation for length check - just concat all message content
    formatted_text = ""
    for msg in messages:
        formatted_text += f"{msg['role']}: {msg['content']}\n"
    return formatted_text


def is_too_generic(text):
    """Check if assistant response is too generic/template-like."""
    generic_phrases = [
        "I cannot help with that",
        "I can't assist with",
        "I'm not able to",
    ]
    
    text_lower = text.lower()
    
    # If response is ONLY a generic phrase (too short)
    if len(text) < 50:
        for phrase in generic_phrases:
            if phrase.lower() in text_lower and len(text) < 100:
                return True
    
    return False


# ============================================================================
# Output and Reporting
# ============================================================================

def save_conversations(conversations, output_dir):
    """
    Save conversations in multiple formats for different uses.
    
    Saves in JSONL format for training and full JSON for analysis.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. JSONL format for training (like gen_synthetic_data.py line 376)
    jsonl_path = output_dir / config.OUTPUT_JSONL
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            # Save just the messages for training
            f.write(json.dumps(conv["messages"]) + '\n')
    
    print(f"✓ Saved {len(conversations)} conversations to {jsonl_path}")
    
    # 2. Full JSON with metadata for analysis
    full_path = output_dir / "safety_conversations_full.json"
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2)
    
    print(f"✓ Saved full data with metadata to {full_path}")
    
    # 3. Debug samples for human inspection
    debug_path = output_dir / config.OUTPUT_DEBUG
    templates.create_debug_output(conversations, debug_path, num_samples=20)
    
    print(f"✓ Saved debug samples to {debug_path}")
    
    # 4. Metadata and statistics
    metadata = create_metadata(conversations)
    metadata_path = output_dir / config.OUTPUT_METADATA
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to {metadata_path}")


def create_metadata(conversations):
    """Create metadata summary with structured information."""
    
    # Count distributions
    by_category = defaultdict(int)
    by_severity = defaultdict(int)
    by_scenario = defaultdict(int)
    by_safety_label = defaultdict(int)
    
    for conv in conversations:
        meta = conv.get("metadata", {})
        by_category[meta.get("category", "unknown")] += 1
        by_severity[meta.get("severity", "unknown")] += 1
        by_scenario[meta.get("scenario_type", "unknown")] += 1
        by_safety_label[meta.get("safety_label", "unknown")] += 1
    
    return {
        "generation_info": {
            "date": datetime.now().isoformat(),
            "generator_model": config.API_CONFIG["model"],
            "total_conversations": len(conversations),
            "config": {
                "target_num": config.TARGET_DISTRIBUTION["num_conversations"],
                "num_workers": config.API_CONFIG["num_workers"],
                "temperature": config.API_CONFIG["temperature"],
            }
        },
        "distribution": {
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
            "by_scenario": dict(by_scenario),
            "by_safety_label": dict(by_safety_label),
        },
        "quality_thresholds": config.QUALITY_THRESHOLDS,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate safety training data for nanochat"
    )
    parser.add_argument(
        "--num_conversations",
        type=int,
        default=config.TARGET_DISTRIBUTION["num_conversations"],
        help="Target number of conversations to generate"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=config.API_CONFIG["num_workers"],
        help="Number of parallel workers for API calls"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.OUTPUT_DIR,
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--overgenerate_factor",
        type=float,
        default=1.3,
        help="Generate extra conversations to account for filtering (e.g., 1.3 = 30% extra)"
    )
    
    args = parser.parse_args()
    
    # Update config
    config.API_CONFIG["num_workers"] = args.num_workers
    
    print("=" * 80)
    print("SAFETY DATA GENERATION")
    print("=" * 80)
    print(f"Target conversations: {args.num_conversations}")
    print(f"Workers: {args.num_workers}")
    print(f"Model: {config.API_CONFIG['model']}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Step 1: Create generation plan
    # Generate extra to account for filtering
    num_to_generate = int(args.num_conversations * args.overgenerate_factor)
    specs = create_generation_plan(num_to_generate)
    
    print(f"\nWill generate {len(specs)} conversations (target {args.num_conversations} after filtering)")
    
    # Step 2: Generate conversations
    print("\n" + "=" * 80)
    print("GENERATION PHASE")
    print("=" * 80)
    raw_conversations = generate_conversation_batch(specs)
    
    print(f"\n✓ Generated {len(raw_conversations)} raw conversations")
    
    # Step 3: Quality filtering
    print("\n" + "=" * 80)
    print("QUALITY FILTERING")
    print("=" * 80)
    filtered_conversations, stats = quality_filter_conversations(raw_conversations)
    
    print(f"\nQuality filter results:")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count}")
    
    print(f"\n✓ Kept {len(filtered_conversations)} conversations after filtering")
    
    # Step 4: Sample to target if we have more than needed
    if len(filtered_conversations) > args.num_conversations:
        print(f"\nSampling {args.num_conversations} from {len(filtered_conversations)} filtered conversations...")
        # Stratified sampling to maintain distribution
        final_conversations = stratified_sample(filtered_conversations, args.num_conversations)
    else:
        final_conversations = filtered_conversations
        if len(final_conversations) < args.num_conversations:
            print(f"\n⚠️  Warning: Only generated {len(final_conversations)} conversations, " 
                  f"target was {args.num_conversations}")
    
    # Step 5: Save output
    print("\n" + "=" * 80)
    print("SAVING OUTPUT")
    print("=" * 80)
    save_conversations(final_conversations, args.output_dir)
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"✓ Generated {len(final_conversations)} safety training conversations")
    print(f"✓ Output saved to {args.output_dir}")
    print("\nNext steps:")
    print("  1. Review debug samples in safety_debug_samples.txt")
    print("  2. Check metadata in safety_metadata.json")
    print("  3. Add safety_conversations.jsonl to scripts/chat_sft.py:")
    print('     CustomJSON(filepath="safety_conversations.jsonl")')
    print("=" * 80)


def stratified_sample(conversations, target_num):
    """Sample conversations while maintaining category distribution."""
    by_category = defaultdict(list)
    for conv in conversations:
        category = conv["metadata"]["category"]
        by_category[category].append(conv)
    
    # Calculate samples per category
    per_category = target_num // len(by_category)
    remainder = target_num % len(by_category)
    
    sampled = []
    for idx, (category, items) in enumerate(sorted(by_category.items())):
        num_samples = per_category + (1 if idx < remainder else 0)
        num_samples = min(num_samples, len(items))
        sampled.extend(random.sample(items, num_samples))
    
    random.shuffle(sampled)
    return sampled


if __name__ == "__main__":
    main()
