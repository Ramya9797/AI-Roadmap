import json
from pydantic import ValidationError
from app.llm import MockLLM
from app.schemas import CopilotResponse

MIN_INTENT_ACCURACY = 90
MIN_ACTION_ACCURACY = 90
MIN_SCHEMA_PASS_RATE = 90
MIN_CITATION_ACCURACY = 90

def load_cases():

    with open("eval/cases.json", "r") as file:
        return json.load(file)


def evaluate():

    cases = load_cases()

    llm = MockLLM()

    total = len(cases)

    intent_correct = 0
    action_correct = 0
    citation_correct = 0
    schema_valid_count = 0

    for case in cases:

        ticket = case["ticket"]

        expected_intent = case["expected_intent"]
        expected_action = case["expected_action"]
        expected_citation = case["expected_citation"]

        result = llm.generate(ticket)

        try:

            validated_result = CopilotResponse.model_validate(result)

            schema_valid = True
            schema_valid_count += 1

            actual_intent = validated_result.intent.value
            actual_action = validated_result.action.value
            actual_citations = validated_result.citations

        except ValidationError as e:

            schema_valid = False

            actual_intent = "INVALID"
            actual_action = "INVALID"
            actual_citations = []

            print("Schema validation failed:")
            print(e)

        intent_match = actual_intent == expected_intent
        action_match = actual_action == expected_action
        citation_match = expected_citation in actual_citations

        if expected_citation:
            citation_match = expected_citation in actual_citations
        else:
            citation_match = len(actual_citations) == 0

        if intent_match:
            intent_correct += 1

        if action_match:
            action_correct += 1

        if citation_match:
            citation_correct += 1

        print()
        print(f"Ticket: {ticket}")
        print(f"Expected intent: {expected_intent}")
        print(f"Actual intent:   {actual_intent}")

        print(f"Expected action: {expected_action}")
        print(f"Actual action:   {actual_action}")
        print(f"Expected citation: {expected_citation}")
        print(f"Actual citations:   {actual_citations}")

        print(f"Schema valid:    {schema_valid}")

        if schema_valid:
            print("Schema: PASS")
        else:
            print("Schema: FAIL")

        if (
            schema_valid
            and intent_match
            and action_match
            and citation_match
        ):
            print("Result: PASS")
        else:
            print("Result: FAIL")

        # if schema_valid and intent_match and action_match and citation_match:
        #     print("Schema: PASS")
        #     print("Result: PASS")
        # else:
        #     print("Schema: FAIL" if not schema_valid else "Schema: PASS")
        #     print("Result: FAIL")

        # actual_intent = result["intent"]
        # actual_action = result["action"]

        # intent_match = actual_intent == expected_intent
        # action_match = actual_action == expected_action

        # if intent_match:
        #     intent_correct += 1

        # if action_match:
        #     action_correct += 1

        # print()
        # print(f"Ticket: {ticket}")
        # print(f"Expected intent: {expected_intent}")
        # print(f"Actual intent:   {actual_intent}")

        # print(f"Expected action: {expected_action}")
        # print(f"Actual action:   {actual_action}")

        # if intent_match and action_match:
        #     print("Result: PASS")
        # else:
        #     print("Result: FAIL")
    # for case in cases:

    #     ticket = case["ticket"]

    #     expected_intent = case["expected_intent"]
    #     expected_action = case["expected_action"]

    #     result = llm.generate(ticket)

    #     actual_intent = result["intent"]
    #     actual_action = result["action"]

    #     if actual_intent == expected_intent:
    #         intent_correct += 1

    #     if actual_action == expected_action:
    #         action_correct += 1

    intent_score = intent_correct / total * 100
    action_score = action_correct / total * 100
    citation_score = citation_correct / total * 100
    schema_score = schema_valid_count / total * 100

    print()
    print(f"Total cases: {total}")
    print(f"Intent accuracy: {intent_score:.2f}%")
    print(f"Action accuracy: {action_score:.2f}%")
    print(f"Citation accuracy: {citation_score:.2f}%")
    print(f"Schema pass rate: {schema_score:.2f}%")




    if (
        intent_score >= MIN_INTENT_ACCURACY
        and action_score >= MIN_ACTION_ACCURACY
        and citation_score >= MIN_CITATION_ACCURACY
        and schema_score >= MIN_SCHEMA_PASS_RATE
    ):
        print("Evaluation Gate: PASS")
    else:
        print("Evaluation Gate: FAIL")
        raise SystemExit(1)

if __name__ == "__main__":
    evaluate()