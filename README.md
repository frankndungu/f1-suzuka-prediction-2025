# Suzuka 2025 F1 Prediction Model (Powered by [Otto.rentals](https://otto.rentals))

This project uses historical Formula 1 data and machine learning to predict the **finishing order** of drivers for the 2025 Japanese Grand Prix at Suzuka. Built to engage data-driven fans, it powers social-ready insights for Otto’s car-loving audience.

---

## What It Does

- Trains a **Random Forest model** using FastF1 race results (2022–2024)
- Factors in:
  - Qualifying (grid position)
  - Track-specific performance at Suzuka
  - Rolling driver form (last 5 races)
  - Team strength & McLaren’s recent surge
  - Experience and rookies’ risk
- Outputs:
  - Full grid predictions with confidence bars
  - Visuals for Instagram/TikTok (podium + full grid)

---

## Requirements

Install all dependencies with:

```bash
pip install fastf1 pandas numpy scikit-learn matplotlib seaborn pillow
```

---

## Outputs

- `suzuka_prediction_full_grid.png`: Full driver ranking with team color codes and error bars
- `suzuka_podium_social.png`: Horizontal podium graphic (for X/IG)
- `suzuka_story_1st.png` ... `3rd.png`: One-card podium slides (for IG Stories or TikTok)

---

## Sample Podium Prediction (as of latest run)

```
 Max Verstappen (Red Bull Racing)
 Lewis Hamilton (Ferrari)
 Charles Leclerc (Ferrari)
```

---

## Run the Model

Clone or open the script and run it:

```bash
python suzuka_f1.py
```

All visual outputs will be saved in the working directory.

---

## Resources

- [FastF1](https://theoehrly.github.io/Fast-F1/) for official race data
- `RandomForestRegressor` from `scikit-learn`
- `matplotlib`, `seaborn`, and `pillow` for visuals
- Driver and team lineup from the official 2025 F1 grid

---

## Author

Frank Ndungu

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Built for [Otto.rentals](https://otto.rentals)

Making car data sexy again.  
Fueling F1 insights for fans who love both speed and stats.
