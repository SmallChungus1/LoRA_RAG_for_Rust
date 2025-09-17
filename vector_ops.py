from main import course_rag
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supporting function calls for Course RAG")
    parser.add_argument("--d", type=bool, help="clear chromadb collections")
    args = parser.parse_args()

    if args.d:
        course_rag_instance = course_rag()
        course_rag_instance.delete_all_chroma_collections()
        print(f"All collections deleted.")
    else:
        print("No action taken. Use --d to delete a collection.")
    