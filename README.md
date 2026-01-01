# TabPFN Deep Dive Project

This repository contains my deep dive into the TabPFN-2.5 ML model.

TabPFN Repository: [https://github.com/PriorLabs/tabPFN](https://github.com/PriorLabs/tabPFN)

## Setup

1. Create virtual environment
2. Install requirements
3. Run `hf auth login` using Hugging Face token for local inference
4. Create `.env` with PRIORLABS_API_KEY for cloud inference
5. Run Local and Cloud smoke tests:
   - `python smoke_test_local.py`
   - `python smoke_test_cloud.py`
