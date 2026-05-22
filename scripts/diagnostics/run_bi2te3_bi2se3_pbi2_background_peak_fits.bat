@echo off
setlocal

python "%~dp0run_all_background_peak_fits.py" ^
  --notebook "%~dp0all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py" ^
  "C:\Users\Kenpo\.local\share\ra_sim\Bi2Te3.json" ^
  "C:\Users\Kenpo\.local\share\ra_sim\Bi2Se3.json" ^
  "C:\Users\Kenpo\.local\share\ra_sim\PbI2.json"
