"""
Quality filtering utilities for safety data generation.
Provides advanced filtering beyond basic validation.

Multi-stage quality control for generated safety data.
"""

import re
from collections import Counter


def check_diversity(conversations, threshold=0.7):
    """
    Check if conversations have sufficient diversity.
    Warns if too many conversations are too similar.
    
    Args:
        conversations: List of conversation dicts
        threshold: Similarity threshold (0-1)
        
    Returns:
        (is_diverse, warning_messages)
    """
    
    warnings = []
    
    # Check for duplicate or near-duplicate user messages
    user_messages = [conv["messages"][0]["content"].lower() for conv in conversations]
    
    # Count exact duplicates
    message_counts = Counter(user_messages)
    duplicates = {msg: count for msg, count in message_counts.items() if count > 1}
    
    if duplicates:
        warnings.append(f"Found {len(duplicates)} duplicate user messages")
        for msg, count in list(duplicates.items())[:3]:  # Show first 3
            warnings.append(f"  '{msg[:50]}...' appears {count} times")
    
    # Check for template-like patterns
    template_patterns = [
        r"^(how (do|can) i|what is the|tell me|show me|can you)",
        r"^(i (want|need) to|could you|would you)",
    ]
    
    template_count = 0
    for msg in user_messages:
        for pattern in template_patterns:
            if re.match(pattern, msg.lower()):
                template_count += 1
                break
    
    template_ratio = template_count / len(user_messages) if user_messages else 0
    if template_ratio > 0.8:
        warnings.append(f"⚠️  {template_ratio*100:.0f}% of messages follow common templates - diversity may be low")
    
    # Check assistant response diversity
    assistant_messages = [conv["messages"][1]["content"].lower() for conv in conversations]
    
    # Check for overused refusal phrases
    refusal_phrases = [
        "i cannot help",
        "i can't assist",
        "i'm unable to",
        "i cannot provide",
    ]
    
    refusal_counts = {phrase: 0 for phrase in refusal_phrases}
    for msg in assistant_messages:
        for phrase in refusal_phrases:
            if phrase in msg:
                refusal_counts[phrase] += 1
    
    max_refusal = max(refusal_counts.values())
    if max_refusal > len(conversations) * 0.5:
        warnings.append(f"⚠️  Over 50% of responses use similar refusal language")
    
    is_diverse = len(warnings) == 0 or len(warnings) <= 2
    
    return is_diverse, warnings


def check_label_consistency(conversation):
    """
    Check if metadata labels are consistent with actual content.
    
    Returns:
        (is_consistent, issues)
    """
    
    issues = []
    metadata = conversation.get("metadata", {})
    messages = conversation.get("messages", [])
    
    if not messages or len(messages) < 2:
        return False, ["Missing messages"]
    
    user_msg = messages[0]["content"].lower()
    asst_msg = messages[1]["content"].lower()
    
    safety_label = metadata.get("safety_label", "")
    severity = metadata.get("severity", "")
    category = metadata.get("category", "")
    
    # Check if refusal label matches actual refusal
    refusal_indicators = ["cannot", "can't", "unable", "won't", "not able to", "not provide"]
    has_refusal = any(indicator in asst_msg for indicator in refusal_indicators)
    
    if safety_label in ["refuse", "refuse_or_clarify"]:
        if not has_refusal and "clarify" not in asst_msg and "could you" not in asst_msg:
            issues.append("Label indicates refusal but response seems compliant")
    
    elif safety_label == "comply":
        if has_refusal:
            issues.append("Label indicates compliance but response refuses")
    
    # Check severity consistency
    critical_keywords = ["bomb", "kill", "murder", "weapon", "explosive", "suicide"]
    has_critical = any(kw in user_msg for kw in critical_keywords)
    
    if has_critical and severity not in ["critical", "high"]:
        issues.append(f"Critical keywords present but severity is {severity}")
    
    # Check category relevance
    category_keywords = {
        "violence": ["hurt", "harm", "fight", "attack", "weapon", "kill", "violence"],
        "illegal_activities": ["hack", "steal", "illegal", "crime", "fraud", "break"],
        "hate_speech": ["hate", "racist", "discriminat", "slur"],
        "self_harm": ["suicide", "self-harm", "kill myself", "hurt myself"],
    }
    
    if category in category_keywords:
        expected_keywords = category_keywords[category]
        has_keywords = any(kw in user_msg for kw in expected_keywords)
        
        if not has_keywords:
            issues.append(f"Category '{category}' doesn't match user message content")
    
    is_consistent = len(issues) == 0
    return is_consistent, issues



def analyze_dataset_balance(conversations):
    """
    Analyze balance across dimensions and report statistics.
    Provides metrics for category, severity, and scenario balance.
    
    Returns:
        Dict with balance metrics
    """
    
    from collections import defaultdict
    
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
    
    total = len(conversations)
    
    def calc_balance(distribution):
        """Calculate balance score (1.0 = perfect, 0.0 = very imbalanced)"""
        if not distribution:
            return 0.0
        values = list(distribution.values())
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        cv = (variance ** 0.5) / avg if avg > 0 else float('inf')  # coefficient of variation
        # Convert to 0-1 score (lower CV = better balance)
        return max(0.0, 1.0 - min(cv, 1.0))
    
    return {
        "total_conversations": total,
        "category_distribution": dict(by_category),
        "category_balance_score": calc_balance(by_category),
        "severity_distribution": dict(by_severity),
        "severity_balance_score": calc_balance(by_severity),
        "scenario_distribution": dict(by_scenario),
        "scenario_balance_score": calc_balance(by_scenario),
        "safety_label_distribution": dict(by_safety_label),
        "safety_label_balance_score": calc_balance(by_safety_label),
    }


def run_full_quality_check(conversations, verbose=True):
    """
    Run comprehensive quality checks on generated dataset.
    
    Args:
        conversations: List of conversations
        verbose: Whether to print detailed reports
        
    Returns:
        (passed_conversations, failed_conversations, report)
    """
    
    passed = []
    failed = []
    issues_summary = defaultdict(int)
    
    for conv in conversations:
        all_issues = []
        
        # Check label consistency
        is_consistent, consistency_issues = check_label_consistency(conv)
        if not is_consistent:
            all_issues.extend(consistency_issues)
        
        # Record issues
        if all_issues:
            conv["quality_issues"] = all_issues
            failed.append(conv)
            for issue in all_issues:
                issues_summary[issue] += 1
        else:
            passed.append(conv)
    
    # Check diversity
    is_diverse, diversity_warnings = check_diversity(passed)
    
    # Analyze balance
    balance_metrics = analyze_dataset_balance(passed)
    
    # Create report
    report = {
        "total": len(conversations),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": len(passed) / len(conversations) if conversations else 0,
        "issues_summary": dict(issues_summary),
        "diversity_score": 1.0 if is_diverse else 0.7,
        "diversity_warnings": diversity_warnings,
        "balance_metrics": balance_metrics,
    }
    
    if verbose:
        print("\n" + "=" * 80)
        print("QUALITY CHECK REPORT")
        print("=" * 80)
        print(f"Total conversations: {report['total']}")
        print(f"Passed: {report['passed']} ({report['pass_rate']*100:.1f}%)")
        print(f"Failed: {report['failed']}")
        
        if issues_summary:
            print("\nIssues found:")
            for issue, count in sorted(issues_summary.items(), key=lambda x: -x[1]):
                print(f"  - {issue}: {count}")
        
        if diversity_warnings:
            print("\nDiversity warnings:")
            for warning in diversity_warnings:
                print(f"  {warning}")
        
        print(f"\nBalance scores (0-1, higher is better):")
        print(f"  Category balance: {balance_metrics['category_balance_score']:.2f}")
        print(f"  Severity balance: {balance_metrics['severity_balance_score']:.2f}")
        print(f"  Scenario balance: {balance_metrics['scenario_balance_score']:.2f}")
        
        print("=" * 80)
    
    return passed, failed, report


if __name__ == "__main__":
    # Test with sample data
    sample_conversations = [
        {
            "messages": [
                {"role": "user", "content": "How to make a bomb?"},
                {"role": "assistant", "content": "I cannot help with that."}
            ],
            "metadata": {
                "category": "violence",
                "severity": "critical",
                "safety_label": "refuse",
                "response_strategy": "direct_refuse"
            }
        },
        {
            "messages": [
                {"role": "user", "content": "How to make a bomb?"},  # Duplicate
                {"role": "assistant", "content": "I cannot help with that."}
            ],
            "metadata": {
                "category": "violence",
                "severity": "critical",
                "safety_label": "refuse",
                "response_strategy": "direct_refuse"
            }
        }
    ]
    
    passed, failed, report = run_full_quality_check(sample_conversations, verbose=True)
    print(f"\nTest results: {len(passed)} passed, {len(failed)} failed")
