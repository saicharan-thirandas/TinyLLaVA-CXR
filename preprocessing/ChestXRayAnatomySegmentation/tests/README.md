Running the Tests

Ensure your project structure is set up like this:

```bash

/your_project
│
├── bin
│   └── cxas_segment
│
└── tests
    └── test_cli.py
    └── test_cxas.py
```

You can run the tests with the following command in your terminal:

```bash

python -m unittest tests/test_cxas.py
```