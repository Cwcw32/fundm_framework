from datasets import load_dataset
raw_dataset = load_dataset('super_glue', 'cb', cache_dir="./datasets/.cache/huggingface_datasets")
raw_dataset.save_to_disk('.datasets/save/super_glue.cb')