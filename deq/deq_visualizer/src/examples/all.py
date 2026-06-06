from pathlib import Path
import importlib.util


def main():
    # Get the current directory
    current_dir = Path(__file__).parent

    # Find all files ending with -library.py
    for file_path in current_dir.glob("*-library.py"):
        # Load the module dynamically
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Call the main function if it exists
        if hasattr(module, "main"):
            print(f"Running {file_path.name}...")
            try:
                module.main()
            except:
                print(f"Error running {file_path.name}")

    # check the deq/deq_runtime/bin folder for additional files and create soft links
    deq_bin_dir = current_dir.parent.parent.parent / "deq_runtime" / "bin"
    dst_path = current_dir / "runtime_bin"
    dst_path.unlink(missing_ok=True)
    print(f"Creating symlink for {dst_path}...")
    try:
        dst_path.symlink_to(deq_bin_dir)
    except Exception as e:
        print(f"Error creating symlink for {dst_path}: {e}")


if __name__ == "__main__":
    main()
