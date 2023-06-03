def fix_jack_glove_memmap_not_found():
    """
    Create memory mapped GloVe embedding required for UCL-NLP model as the link to download it is broken
    """
    from pathlib import Path

    import numpy as np
    from jack.io.embeddings import load_embeddings
    from jack.io.embeddings.memory_map import save_as_memory_map_dir, load_memory_map_dir

    memmap_glove_path = Path("../jack/data/GloVe/glove.840B.300d.memory_map_dir")
    if memmap_glove_path.exists():
        print(f"{memmap_glove_path.resolve()} exists! Skipping...")
        return
    else:
        print(f"Creating memory mapped GloVe: {memmap_glove_path.resolve()}")
    
    embeddings = load_embeddings(memmap_glove_path.parent.joinpath("glove.840B.300d.txt").resolve().as_posix(), "glove")
    save_as_memory_map_dir(memmap_glove_path.resolve().as_posix(), embeddings)
    loaded_embeddings = load_memory_map_dir(memmap_glove_path.resolve().as_posix())\

if __name__ == "__main__":
    fix_jack_glove_memmap_not_found()