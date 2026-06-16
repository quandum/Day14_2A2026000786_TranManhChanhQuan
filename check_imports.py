"""
Check what's available in ragas and trulens packages.
"""
import os

# === Check RAGAS ===
print("=" * 60)
print("1. RAGAS")
print("=" * 60)

try:
    import ragas
    print(f"  Version: {ragas.__version__}")
    print(f"  Path: {ragas.__file__}")
except Exception as e:
    print(f"  Import error: {e}")

try:
    from ragas.metrics.collections import faithfulness, answer_relevancy, context_recall, context_precision
    print("  ✅ metrics.collections: faithfulness, answer_relevancy, context_recall, context_precision")
except Exception as e:
    print(f"  ❌ metrics.collections: {e}")

try:
    from ragas.evaluation import evaluate
    print("  ✅ evaluation.evaluate")
except Exception as e:
    print(f"  ❌ evaluation.evaluate: {e}")

try:
    from ragas import EvaluationDataset
    print("  ✅ EvaluationDataset")
except Exception as e:
    print(f"  ❌ EvaluationDataset: {e}")

try:
    from ragas.llms import llm_factory
    print("  ✅ llms.llm_factory")
except Exception as e:
    print(f"  ❌ llms.llm_factory: {e}")


# === Check TruLens ===
print("\n" + "=" * 60)
print("2. TruLens")
print("=" * 60)

try:
    import trulens
    print(f"  Path: {trulens.__file__}")
except Exception as e:
    print(f"  Import error: {e}")

try:
    import trulens.core
    from trulens.core import Feedback
    print(f"  ✅ trulens.core.Feedback")
except Exception as e:
    print(f"  ❌ trulens.core.Feedback: {e}")

try:
    import trulens.feedback
    print(f"  ✅ trulens.feedback")
    # List contents
    feedback_dir = os.path.dirname(trulens.feedback.__file__)
    print(f"  Files in trulens.feedback:")
    for f in sorted(os.listdir(feedback_dir)):
        if f.endswith('.py') and not f.startswith('_'):
            print(f"    - {f}")
except Exception as e:
    print(f"  ❌ trulens.feedback: {e}")

try:
    from trulens.feedback.provider.openai import OpenAI
    print("  ✅ trulens.feedback.provider.openai.OpenAI")
except Exception as e:
    print(f"  ❌ trulens.feedback.provider.openai.OpenAI: {e}")

try:
    from trulens.core.feedback.feedback import Feedback as CoreFeedback
    print("  ✅ trulens.core.feedback.feedback.Feedback")
except Exception as e:
    print(f"  ❌ trulens.core.feedback.feedback.Feedback: {e}")

try:
    from trulens_eval import Feedback as TrulensEvalFeedback
    print("  ✅ trulens_eval.Feedback")
except Exception as e:
    print(f"  ❌ trulens_eval.Feedback: {e}")


# === Check GOOGLE_API_KEY ===
print("\n" + "=" * 60)
print("3. Environment")
print("=" * 60)
print(f"  GOOGLE_API_KEY: {'✅ SET' if os.environ.get('GOOGLE_API_KEY') else '❌ NOT SET'}")
print(f"  OPENAI_API_KEY: {'✅ SET' if os.environ.get('OPENAI_API_KEY') else '❌ NOT SET'}")