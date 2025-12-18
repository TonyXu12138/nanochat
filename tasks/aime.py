"""
AIME (American Invitational Mathematics Examination) evaluation.
AIME24 and AIME25 datasets.
"""

from datasets import load_dataset, concatenate_datasets
from tasks.common import Task

class AIME(Task):
    
    def __init__(self, subset, split, num_fewshot=0, fewshot_examples=None, **kwargs):
        super().__init__(**kwargs)
        # subset should be "aime24" or "aime25"
        assert subset in ["aime24", "aime25"], "AIME subset must be aime24|aime25"
        assert split in ["train", "test"], "AIME split must be train|test"
        self.subset = subset
        self.num_fewshot = num_fewshot
        
        if subset == "aime24":
            self.ds = load_dataset("Maxwell-Jia/AIME_2024", split='train').shuffle(seed=42)
        elif subset == "aime25":
            ds_part1 = load_dataset("opencompass/AIME2025", 'AIME2025-I', split='test')
            ds_part2 = load_dataset("opencompass/AIME2025", 'AIME2025-II', split='test')
            # Concatenate the two parts and then shuffle
            self.ds = concatenate_datasets([ds_part1, ds_part2]).shuffle(seed=42)

        # Few-shot examples
        if fewshot_examples is not None:
            self.fewshot_examples = fewshot_examples
        else:
            # Default Standard hard math problems
            self.fewshot_examples = [
                {
                    "problem": "Find the sum of all integers $x$ such that $|x-3| + |x+5| < 12$.",
                    "solution": "We check three cases.\nCase 1: $x < -5$. Then $|x-3| = 3-x$ and $|x+5| = -x-5$. The inequality becomes $3-x - x - 5 < 12$, so $-2x - 2 < 12$, $-2x < 14$, $x > -7$. The integers are $-6$.\nCase 2: $-5 \\le x \\le 3$. Then $|x-3| = 3-x$ and $|x+5| = x+5$. The inequality becomes $3-x + x+5 < 12$, so $8 < 12$, which is always true. The integers are $-5, -4, -3, -2, -1, 0, 1, 2, 3$.\nCase 3: $x > 3$. Then $|x-3| = x-3$ and $|x+5| = x+5$. The inequality becomes $x-3 + x+5 < 12$, so $2x + 2 < 12$, $2x < 10$, $x < 5$. The integer is $4$.\nThe sum of the integers is $(-6) + (-5) + (-4) + (-3) + (-2) + (-1) + 0 + 1 + 2 + 3 + 4 = -11$.",
                    "answer": "-11"
                },
                {
                    "problem": "Let $P(x)$ be a polynomial such that $P(x) = P(0) + P(1)x + P(2)x^2$ and $P(-1) = 1$. Find $P(3)$.",
                    "solution": "Let $P(x) = c + bx + ax^2$. Then $P(0) = c$, $P(1) = a+b+c$, $P(2) = 4a+2b+c$. The given equation is $c + bx + ax^2 = c + (a+b+c)x + (4a+2b+c)x^2$. Comparing coefficients, we have $a = 4a+2b+c$ and $b = a+b+c$. The second equation gives $a+c=0$, so $c=-a$. The first equation gives $3a+2b+c=0$, so $3a+2b-a=0$, $2a+2b=0$, $b=-a$. Thus $P(x) = -a - ax + ax^2 = a(x^2-x-1)$. We are given $P(-1)=1$, so $a(1+1-1)=1$, $a=1$. Thus $P(x) = x^2-x-1$. We want $P(3) = 9-3-1 = 5$.",
                    "answer": "5"
                },
                {
                    "problem": "Determine the number of ordered pairs $(a, b)$ of positive integers such that $a \\le 10$ and $b \\le 10$ and $a+b$ is divisible by 3.",
                    "solution": "We want $a+b \\equiv 0 \\pmod{3}$.\nIf $a \\equiv 1 \\pmod{3}$ (i.e., $a \\in \\{1, 4, 7, 10\\}$, 4 choices), then we need $b \\equiv 2 \\pmod{3}$ (i.e., $b \\in \\{2, 5, 8\\}$, 3 choices). This gives $4 \\times 3 = 12$ pairs.\nIf $a \\equiv 2 \\pmod{3}$ (i.e., $a \\in \\{2, 5, 8\\}$, 3 choices), then we need $b \\equiv 1 \\pmod{3}$ (i.e., $b \\in \\{1, 4, 7, 10\\}$, 4 choices). This gives $3 \\times 4 = 12$ pairs.\nIf $a \\equiv 0 \\pmod{3}$ (i.e., $a \\in \\{3, 6, 9\\}$, 3 choices), then we need $b \\equiv 0 \\pmod{3}$ (i.e., $b \\in \\{3, 6, 9\\}$, 3 choices). This gives $3 \\times 3 = 9$ pairs.\nTotal number of pairs is $12 + 12 + 9 = 33$.",
                    "answer": "33"
                }
            ]

        # Option 2: If data is in local JSON files
        # import json
        # with open(f"data/aime/{subset}_{split}.json", 'r') as f:
        #     self.ds = json.load(f)
    
    @property
    def eval_type(self):
        # AIME problems are math questions with numeric answers, use generative evaluation
        return 'generative'
    
    def num_examples(self):
        return len(self.ds)
    
    def get_example(self, index):
        """
        Get a single AIME problem with prompt engineering (Zero-shot or Few-shot).
        """
        row = self.ds[index]
        # Handle different column naming conventions (Problem/Answer vs problem/answer)
        problem = row.get('problem') or row.get('Problem')
        answer = str(row.get('answer') or row.get('Answer'))
        
        # Prepare the prompt content
        # OpenCompass format: "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        prompt_suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
        
        final_prompt = ""
        
        # Add Few-shot examples if requested
        if self.num_fewshot > 0:
            examples = self.fewshot_examples[:self.num_fewshot]
            for ex in examples:
                final_prompt += f"Problem: {ex['problem']}{prompt_suffix}\nSolution: {ex['solution']}\nFinal Answer: \\boxed{{{ex['answer']}}}\n\n"
        
        # Add the actual problem
        final_prompt += f"Problem: {problem}{prompt_suffix}\n"
        
        # Create conversation
        messages = [
            {"role": "user", "content": final_prompt},
            {"role": "assistant", "content": answer}
        ]
        conversation = {
            "messages": messages,
            "reference_answer": answer # Store for easier access
        }
        return conversation
    
    def evaluate(self, conversation, assistant_response):
        """
        Evaluate AIME answer using robust math equivalence.
        """
        assert isinstance(assistant_response, str), "Expecting string response"
        
        # 1. Get Reference Answer
        correct_answer = conversation.get('reference_answer')
        if correct_answer is None:
            assistant_message = conversation['messages'][-1]
            assert assistant_message['role'] == "assistant"
            correct_answer = assistant_message['content']
        
        # 2. Extract Predicted Answer
        predicted_answer = self.extract_answer(assistant_response)
        if predicted_answer is None:
            return 0
            
        # 3. Compare using robust equivalence check
        return int(self.is_equiv(predicted_answer, correct_answer))

    def is_equiv(self, pred, ref):
        """
        Check if two answers are mathematically equivalent.
        Strategies:
        1. Exact string match (normalized)
        2. Float/Integer comparison (handles 42.0 == 42)
        3. SymPy simplification (handles fractions/expressions)
        """
        # Normalize strings first
        pred_str = str(pred).strip().replace(",", "").replace("$", "")
        ref_str = str(ref).strip().replace(",", "").replace("$", "")
        
        # Strategy 1: Exact String Match
        if pred_str == ref_str:
            return True
            
        # Strategy 2: Numerical Comparison (covers 042, 42.0, 42)
        try:
            pred_val = float(pred_str)
            ref_val = float(ref_str)
            # Check for float equality with tolerance
            if abs(pred_val - ref_val) < 1e-6:
                return True
        except ValueError:
            pass

        # Strategy 3: SymPy Symbolic Comparison (Most robust)
        try:
            from sympy import simplify, parse_expr
            
            def parse_math(s):
                # Try latex2sympy2 first if available (for \frac, \sqrt, etc.)
                try:
                    from latex2sympy2 import latex2sympy
                    try:
                        return latex2sympy(s)
                    except Exception:
                        print('latex2sympy2 failed, fallback to standard sympy parsing')
                        pass
                except ImportError:
                    print('latex2sympy2 not installed, fallback to standard sympy parsing')
                    pass
                
                # Fallback to standard sympy parsing (for simple expressions like "42", "x+1")
                # Pre-process basic latex that sympy can't handle but is easy to fix
                s = s.replace(r'\pi', 'pi').replace('^', '**')
                return parse_expr(s)

            # Calculate difference
            diff = simplify(parse_math(pred_str) - parse_math(ref_str))
            if diff == 0:
                return True
                
        except ImportError:
            # Sympy not installed, fallback strictly to numeric strategies above
            print('sympy not installed, fallback strictly to numeric strategies above')
            pass
        except Exception:
            # Sympy parsing failed (e.g. invalid syntax)
            print('sympy parsing failed, fallback strictly to numeric strategies above')
            pass
            
        return False
    
    @staticmethod
    def extract_answer(text):
        """
        Extract answer prioritizing \\boxed{}. 
        Robustly handles LaTeX spacing and nested braces (simple cases).
        """
        import re
        
        # 1. Try to extract \boxed{} content
        # Matches \boxed{...} allowing for whitespace between \boxed and {
        # The content inside can be anything except }
        boxed_matches = re.findall(r'\\boxed\s*\{([^}]+)\}', text)
        if boxed_matches:
            return boxed_matches[-1].strip()
        
        # 2. Fallback: Extract last number
        # Enhanced regex to handle:
        # - Integers: 42
        # - Floats: 42.0
        # - Commas: 1,000
        # This regex looks for numbers that might contain commas or decimals
        # It avoids matching versions like 1.2.3
        
        # Remove commas for easier extraction logic if needed, but here we want to extract first
        # Pattern: digits, optionally followed by (dot/comma + digits)
        # We search for the last occurrence
        # matches: 42, 42.0, 1,000, 0.5
        matches = re.findall(r'-?\d+(?:[,\.]\d+)*', text)
        if matches:
            # Return the last number found, stripping commas if present so it can be parsed later
            return matches[-1]
        
        return None
    
    def reward(self, conversation, assistant_response):
        """Used for RL training"""
        is_correct = self.evaluate(conversation, assistant_response)
        return float(is_correct)

if __name__ == "__main__":
    print("=== Testing AIME Task ===")
    
    # 1. Test Evaluation Logic (Unit Tests)
    print("\n[Test 1] Evaluation Logic")
    # Initialize task with dummy data just to access methods
    # We use a try-catch for init in case dataset download fails, but we want to test eval logic regardless
    try:
        task = AIME(subset="aime24", split="train")
    except Exception:
        # If dataset load fails, we can still test eval logic by mocking the class or just instantiation
        # Here we assume instantiation might fail only on dataset load
        print("Warning: Could not load dataset, creating bare instance for eval testing")
        task = AIME.__new__(AIME) # Bypass __init__
    
    test_cases = [
        # (Prediction, Reference, Expected, Description)
        ("42", "42", 1, "Exact match"),
        ("The answer is \\boxed{42}", "42", 1, "Standard boxed"),
        ("Result: \\boxed{ 42 }", "42", 1, "Boxed with spaces"),
        ("42.0", "42", 1, "Float equivalence (Model output)"),
        ("042", "42", 1, "Leading zero (Model output)"),
        ("1,000", "1000", 1, "Comma handling"),
        ("Answer is 42", "42", 1, "Fallback extraction"),
        ("It is 41", "42", 0, "Wrong answer"),
        ("Values 10 and 42", "42", 1, "Last number fallback"),
        # AIME answers are always integers, so we focus on integer equivalence
        ("\\boxed{999.000}", "999", 1, "High precision float"),
        ("007", "7", 1, "James Bond integer"),
    ]
    
    for pred, ref, exp, desc in test_cases:
        # Mock conversation dict
        conv = {'reference_answer': ref}
        try:
            score = task.evaluate(conv, pred)
            status = "PASS" if score == exp else "FAIL"
            print(f"{status:<4} | {desc:<30} | Pred: {pred:<20} | Ref: {ref} -> Got {score}")
        except Exception as e:
            print(f"ERR  | {desc:<30} | {e}")

    # 2. Test Data Loading (Integration Test)
    print("\n[Test 2] Data Loading & Prompting")
    try:
        task = AIME(subset="aime24", split="train", num_fewshot=1)
        print(f"Successfully loaded AIME24. Size: {len(task.ds)}")
        
        ex = task.get_example(0)
        prompt = ex['messages'][0]['content']
        ref = ex['reference_answer']
        
        print(f"\nReference Answer: {ref}")
        print("-" * 40)
        print(f"Prompt Preview:\n{prompt[:300]}...\n[...]\n{prompt[-200:]}")
        print("-" * 40)
        
        # Verify format
        if "Problem:" in prompt and "\\boxed{}" in prompt:
            print("PASS | Prompt format check (OpenCompass style)")
        else:
            print("FAIL | Prompt format check")
            
        if "Solution:" in prompt:
             print("PASS | Few-shot inclusion check")
        else:
             print("FAIL | Few-shot inclusion check")
             
    except Exception as e:
        print(f"SKIP | Data loading test failed (Network/Dataset issue): {e}")