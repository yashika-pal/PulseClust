# Radar Project

PulseClust is a web-based tool for analyzing radar pulse data and clustering them by their likely source — such as aircraft — using unsupervised machine learning. It leverages HDBSCAN to group radar pulses based on features like:

- Time of Arrival (TOA)
- Pulse Radar Interval
- Amplitude
- Frequency
- Pulse Width

## File Structure

- `app.py` — main Flask app
- `static/` — CSS files
- `templates/` — HTML templates
- `*.joblib` — trained model and scaler files
