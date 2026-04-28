def generate_explanation(metrics):
    dp = metrics["demographic_parity_difference"]
    eo = metrics["equalized_odds_difference"]

    explanation = ""

    # Demographic parity interpretation
    if abs(dp) < 0.1:
        explanation += "Demographic parity difference is low, indicating similar selection rates across groups.\n"
    else:
        explanation += "There is a noticeable disparity in selection rates across groups.\n"

    # Equalized odds interpretation
    if abs(eo) < 0.1:
        explanation += "Model error rates are relatively balanced across groups.\n"
    else:
        explanation += "Model shows unequal error rates across groups.\n"

    explanation += "\nSuggested Actions:\n"
    explanation += "- Check data imbalance\n"
    explanation += "- Consider removing or transforming sensitive features\n"
    explanation += "- Use fairness-aware algorithms\n"

    return explanation