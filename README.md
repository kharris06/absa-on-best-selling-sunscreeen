# Fine-Tuning InstructABSA on Best-Selling Sunscreen Reviews

This repository contains a fine-tuned version of the [InstructABSA](https://github.com/kevinscaria/InstructABSA) model, customized to perform Aspect-Based Sentiment Analysis (ABSA) on top-selling sunscreen product reviews in the Indonesian language.

## 🔍 Project Overview

Aspect-Based Sentiment Analysis (ABSA) allows us to detect sentiment toward specific aspects (e.g., texture, whitecast, scent, oiliness, price) within a product review. This project focuses on:

- Using the **InstructABSA** architecture (based on T5)
- Fine-tuning the model on a custom dataset consisting of Indonesian sunscreen reviews
- Extracting **aspect terms** and their associated **sentiments** (positive, neutral, negative)

---

## 🧠 Base Model

This project uses the original [InstructABSA](https://github.com/kevinscaria/InstructABSA) repository developed by Kevin Scaria. The original repo provides a framework for instruction-based ABSA using T5 models.



