# RA Simulation

This project contains a diffraction simulation tool with a Tk based GUI.

## Running

After installing the required packages (``pip install -r requirements.txt`` or the
packages listed in ``setup.py``) launch the GUI with:

```bash
python main.py
```


## Troubleshooting


Set the environment variable ``RA_SIM_DEBUG`` to ``1`` (or any truthy value) to
print a summary of these arrays via ``ra_sim.debug_utils.check_ht_arrays``.  On
Linux/macOS use ``export RA_SIM_DEBUG=1``; in Windows ``cmd`` use
``set RA_SIM_DEBUG=1`` or in PowerShell ``$env:RA_SIM_DEBUG='1'``.  You can also
call ``ra_sim.debug_utils.debug_print`` in your own code to emit messages only
when debug mode is active.  For manual inspection insert the snippet below in

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
