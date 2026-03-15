# Dataset: Enron Email Slice

## Source

- **Dataset:** [Enron Email Dataset (wcukierski)](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)
- **License:** Public domain / open for research (FERC release).

## How to add .eml files to `data/emails/`

1. **Download the dataset from Kaggle**
   - Sign in at [Kaggle](https://www.kaggle.com)
   - Open: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
   - Click **Download** and unzip the archive.

2. **Copy `emails.csv` into the project root**
   - From the unzipped folder, copy `emails.csv` to:
   - `E:\email-rag\emails.csv` (same folder as `slice_data.py` and `ingest.py`).

3. **Create the slice and write .eml files**
   ```powershell
   cd E:\email-rag
   venv\Scripts\activate
   python slice_data.py
   ```
   This writes a subset of emails (e.g. from `allen-p/`) as `.eml` files into `data/emails/`.

4. **Index the emails**
   ```powershell
   python ingest.py
   ```

## Slice selection

- **Source:** Kaggle Enron CSV (`emails.csv`) with columns `file`, `message`.
- **Selection:** One mailbox prefix (e.g. `allen-p/`), first N messages (e.g. 250) to keep the slice small (100–300 messages, 10–20+ threads).
- **Output:** `.eml` files in `data/emails/` for use by `ingest.py`.

## Final counts (after you run the slice)

After running `slice_data.py` and `ingest.py`, fill in:

- **Threads:** (from ingest output)
- **Messages:** (from ingest output)
- **Attachments:** (if any PDFs in the slice; Enron CSV may have few)
- **Approx. text size:** (from index size / chunk count)

## Preprocessing

- Raw `message` column from CSV is written as-is to `.eml`.
- `slice_data.py` skips empty messages and uses a hash of `file` for the filename to avoid collisions.
