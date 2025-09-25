## Getting Started

Follow the steps below to set up and run the application.

### Prerequisites

- Python 3.x
- pip

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/username/repository.git
    ```

2. Navigate to the project directory:

    ```bash
    cd repository
    ```

3. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:

    ```bash
    source venv/bin/activate
    ```

5. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Run the main application:

    ```bash
    python main.py
    ```

2. Run the data update application:

    > Note: The data update checks for duplicates. The folders are organized by class and the image files are named after the student IDs.

    ```bash
    python importApp.py
    ```
### Build window app
1. we can use tool auto-py-to-exe to build 
     ```bash
     pip install auto-py-to-exe && auto-py-to-exe
     ```

### Contributing

Contributions are welcome! Please follow the [contribution guidelines](CONTRIBUTING.md).

### License

This project is licensed under the [MIT License](LICENSE).
