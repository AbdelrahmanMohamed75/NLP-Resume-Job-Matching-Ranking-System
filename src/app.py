from model import load_model


def main():
    model, features = load_model()

    print("Resume-Job Matching System")
    print("---------------------------")
    print("Model loaded successfully.")
    print(f"Number of features: {len(features)}")


if __name__ == "__main__":
    main()
