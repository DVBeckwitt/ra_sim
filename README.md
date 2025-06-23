# RA Simulation

This project contains a diffraction simulation tool with a Tk based GUI.

## Running

After installing the required packages (``pip install -r requirements.txt`` or the
packages listed in ``setup.py``) launch the GUI with:

```bash
python main.py
```

## Troubleshooting

If the program prints ``Loaded saved profile ...`` and then exits without
showing the GUI, the parameters passed to the Numba compiled
``process_peaks_parallel`` routine may be malformed.  When using fractional ``L``
values the ``miller1`` and ``intens1`` arrays should contain floating point
numbers and be contiguous.  You can insert the following diagnostic snippet in
``main.py`` right after ``ht_dict_to_arrays`` is called:

```python
print('miller1 dtype:', miller1.dtype, 'shape:', miller1.shape)
print('L range:', miller1[:, 2].min(), miller1[:, 2].max())
print('intens1 dtype:', intens1.dtype, 'min:', intens1.min(), 'max:', intens1.max())
print('miller1 contiguous:', miller1.flags['C_CONTIGUOUS'])
print('intens1 contiguous:', intens1.flags['C_CONTIGUOUS'])
```

Run the script from the command line to see the output.  If the arrays look
reasonable, enable the debug simulation (``Run Debug Simulation`` button) or set
the environment variable ``RA_SIM_DEBUG=1`` (or change ``DEBUG_ENABLED`` to
``True`` in ``main.py``) to write detailed logs from ``diffraction_debug.py``.
These logs show intersection calculations for each reflection and help pinpoint
where processing stops.

Finally, running the unit tests provides a quick check that the intensity helper
is functioning correctly:

```bash
pytest
```
