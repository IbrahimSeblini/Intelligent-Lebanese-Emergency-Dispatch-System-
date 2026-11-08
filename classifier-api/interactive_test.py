import argparse

from hybrid_emergency import (
    build_missing_fields_prompt,
    classify_with_details,
    get_zero_shot_classifier,
)

zero_shot_classifier = get_zero_shot_classifier()


def classify_emergency(
    text: str,
    threshold: float,
    force_label: bool,
    caller_phone: str | None,
):
    """Direct pretrained classification + contact extraction."""
    return classify_with_details(
        text,
        classifier=zero_shot_classifier,
        threshold=threshold,
        force_label=force_label,
        fallback_phone=caller_phone,
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Uncertain threshold (used only when --allow-uncertain is set).",
    )
    parser.add_argument(
        "--allow-uncertain",
        action="store_true",
        help="Allow uncertain results (default behavior is force classification).",
    )
    parser.add_argument(
        "--caller-phone",
        default=None,
        help="Phone number of the caller/session (used if no phone appears in text).",
    )
    args = parser.parse_args()

    while True:
        msg = input("Enter an emergency message (or type 'exit'): ")
        if msg.lower() == "exit":
            break

        result = classify_emergency(
            msg,
            threshold=args.threshold,
            force_label=not args.allow_uncertain,
            caller_phone=args.caller_phone,
        )

        prompt = build_missing_fields_prompt(result["missing_fields"])
        if prompt:
            print(prompt)

        if "name" in result["missing_fields"]:
            while True:
                name = input("Name: ").strip()
                if name:
                    result["contact"]["name"] = name
                    break
            result["missing_fields"] = [x for x in result["missing_fields"] if x != "name"]

        if "location" in result["missing_fields"]:
            while True:
                location = input("Location: ").strip()
                if location:
                    result["contact"]["location"] = location
                    break
            result["missing_fields"] = [x for x in result["missing_fields"] if x != "location"]

        print("Result:", result)
        print()


if __name__ == "__main__":
    main()