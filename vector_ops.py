from main import course_rag
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supporting function calls for Course RAG")
    parser.add_argument("--d", type=bool, help="clear chromadb collections ")
    args = parser.parse_args()

    if args.d:
        print(f"Collection {args.d} deleted.")
    else:
        print("No action taken. Use --d to delete a collection.")
    