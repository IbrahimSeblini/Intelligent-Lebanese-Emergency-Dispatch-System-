import argparse

from hybrid_emergency import classify_emergency_zero_shot, get_zero_shot_classifier

zero_shot_classifier = get_zero_shot_classifier()


def classify_emergency(text: str, threshold: float, force_label: bool):
    return classify_emergency_zero_shot(
        text,
        classifier=zero_shot_classifier,
        uncertain_threshold=threshold,
        force_label=force_label,
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
    args = parser.parse_args()

    examples = [
        "there is a fire in the building",
        "my father collapsed and is not breathing",
        "someone is trying to break into my house",
        "help please come fast",
        "something happened and I am scared",
    ]

    for msg in examples:
        result = classify_emergency(
            msg,
            threshold=args.threshold,
            force_label=not args.allow_uncertain,
        )
        print("Message:", msg)
        print("Prediction:", result)
        print("-" * 40)


if __name__ == "__main__":
    main()