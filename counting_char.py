def count_characters_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            character_count = len(content)
            print(f"Total characters in '{file_path}': {character_count}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")

# Example usage:
if __name__ == "__main__":
    file_path = input("Enter the path of the file to count characters: ")
    count_characters_in_file(file_path)
