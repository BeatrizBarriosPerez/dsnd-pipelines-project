# StyleSense: Automated Review Analysis Pipeline

StyleSense is a data pipeline project that processes and analyzes customer reviews for fashion products. The pipeline ingests raw review data, performs cleaning and transformation, and generates insights to help understand customer sentiment and product feedback. The project demonstrates best practices in data engineering, including reproducible pipelines, modular code, and clear documentation.

## Getting Started

Instructions for how to get a copy of the project running on your local machine.

### Dependencies

This project requires Python 3.8 or higher and the following Python packages:

```
pandas>=1.3.0
scikit-learn>=1.0.0
spacy>=3.0.0
notebook>=6.4.0
pytest>=7.0.0 (for testing)
```

Additionally, you'll need to download the spaCy English language model:

```bash
python -m spacy download en_core_web_sm
```

### Installation

Follow these steps to get a development environment running:

1. **Clone the repository**

```bash
git clone <your-fork-url>
cd dsnd-pipelines-project
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**

```bash
pip install -r requirements.txt
```

4. **Download the spaCy language model**

```bash
python -m spacy download en_core_web_sm
```

5. **Launch Jupyter Notebook**

```bash
jupyter notebook
```

Then navigate to `StyleSense/StyleSense_Pipeline.ipynb` to view and run the pipeline.

## Testing

The project includes a comprehensive test suite to verify the pipeline functionality.

### Running Tests

To run all tests, execute the following from the project root directory:

```bash
pytest tests/ -v
```

For a specific test file:

```bash
pytest tests/test_pipeline.py -v
```

### Break Down Tests

The test suite is organized into three main test classes:

**TestDataLoading**
- `test_data_file_exists`: Verifies that the reviews dataset file exists
- `test_data_loads_successfully`: Ensures data can be loaded without errors
- `test_data_has_expected_columns`: Checks all required columns are present
- `test_data_has_no_missing_values`: Validates data completeness
- `test_target_column_is_binary`: Confirms target variable is properly encoded (0/1)

**TestFeatureTransformations**
- `test_numerical_pipeline`: Verifies numerical features (Age, Positive Feedback Count) are properly scaled
- `test_categorical_pipeline`: Ensures categorical features are one-hot encoded correctly
- `test_text_vectorization`: Validates TF-IDF vectorization of review text

**TestPipelineTraining**
- `test_pipeline_fits_without_error`: Confirms the complete pipeline trains successfully
- `test_model_can_predict`: Verifies the trained model can make predictions
- `test_model_accuracy_above_baseline`: Ensures model performance exceeds random baseline (50%)

These tests ensure the data pipeline is working correctly and the model is learning meaningful patterns from the data.

## Project Instructions

### Using the Pipeline

1. **Explore the Data**: Open `StyleSense/StyleSense_Pipeline.ipynb` and review the data exploration section to understand the dataset structure and characteristics.

2. **Run the Pipeline**: Execute the notebook cells sequentially to:
   - Load the fashion review dataset
   - Prepare features (numerical, categorical, and text)
   - Build preprocessing pipelines for each feature type
   - Train a RandomForestClassifier to predict product recommendations
   - Evaluate model performance

3. **Customize the Model**: The pipeline is modular and can be extended:
   - Add new feature engineering steps
   - Experiment with different text preprocessing techniques
   - Try alternative classification algorithms
   - Tune hyperparameters for better performance

4. **Run Tests**: Verify your changes don't break the pipeline by running the test suite:
   ```bash
   pytest tests/ -v
   ```

### Key Files

- `StyleSense/StyleSense_Pipeline.ipynb`: Main Jupyter notebook with the complete ML pipeline
- `StyleSense/data/reviews.csv`: Fashion product reviews dataset (18,442 reviews)
- `tests/test_pipeline.py`: Automated tests for pipeline validation
- `requirements.txt`: Python package dependencies

## Built With

* [Python](https://www.python.org/) - Programming language
* [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
* [scikit-learn](https://scikit-learn.org/) - Machine learning library for pipeline construction and modeling
* [spaCy](https://spacy.io/) - Natural language processing for text feature extraction
* [Jupyter Notebook](https://jupyter.org/) - Interactive development environment
* [pytest](https://pytest.org/) - Testing framework

## License

[License](LICENSE.txt)
