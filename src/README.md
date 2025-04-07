# hVOS Readout Project

This project computes the hVOS readout from a biological circuit infused with VSD. It takes into account the morphologies of cells and their corresponding voltage traces to generate a readout that can be visualized.

## Project Structure

```
vsd-readout-project
├── src
│   ├── hVOS
│   │   ├── __init__.py       # Marks the vsd directory as a package
│   │   ├── cell.py           # Defines the Cell class
│   │   ├── morphology.py      # Defines the Morphology class
│   │   ├── voltage_trace.py   # Defines the VoltageTrace class
|   |   ├── camera.py          # Define the Camera class
│   │   └── hvos_readout.py     # Defines the hVOSReadout class
├── requirements.txt           # Lists project dependencies
└── README.md                  # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
