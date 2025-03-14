# VSD Readout Project

This project computes the voltage sensitive dye (VSD) readout from a biological circuit infused with VSD. It takes into account the morphologies of cells and their corresponding voltage traces to generate a readout that can be visualized.

## Project Structure

```
vsd-readout-project
├── src
│   ├── main.py               # Entry point of the application
│   ├── vsd
│   │   ├── __init__.py       # Marks the vsd directory as a package
│   │   ├── cell.py           # Defines the Cell class
│   │   ├── morphology.py      # Defines the Morphology class
│   │   ├── voltage_trace.py   # Defines the VoltageTrace class
│   │   └── vsd_readout.py     # Defines the VSDReadout class
├── requirements.txt           # Lists project dependencies
└── README.md                  # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the application, execute the following command:

```bash
python src/main.py
```

Make sure to have your cell data, morphologies, and voltage traces prepared and accessible to the application.

## Classes Overview

- **Cell**: Represents a biological cell and includes methods to retrieve its morphology and voltage traces.
- **Morphology**: Represents the structure of a cell and includes methods to load and manipulate morphology data.
- **VoltageTrace**: Represents the voltage traces for each compartment of a cell and includes methods to load and process voltage data.
- **VSDReadout**: Computes the VSD readout based on the provided cell morphologies and voltage traces, generating output for visualization.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.