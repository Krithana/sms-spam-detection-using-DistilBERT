import pandas as pd
import matplotlib.pyplot as plt

# Grab the results file generated after adversarial testing
# (Assumes this CSV has columns: Changed, Original Text, etc.)
data = pd.read_csv("adversarial_results.csv")

# Count how many predictions were stable vs. flipped
# Note: 'Changed' should have values like 'Yes' or 'No'
num_flipped = data[data['Changed'] == 'Yes'].shape[0]
num_unchanged = data[data['Changed'] == 'No'].shape[0]

# --- Create a basic bar chart ---
plt.figure(figsize=(6, 4))
categories = ['Unchanged', 'Flipped']
counts = [num_unchanged, num_flipped]
colors = ['lightgreen', 'red']

# Honestly just a simple visualization to get a sense of change frequency
plt.bar(categories, counts, color=colors)
plt.title('Model Prediction Change After Adversarial Attacks')
plt.ylabel('Number of Samples')
plt.tight_layout()  # Not always necessary, but helps if labels get clipped
plt.savefig("bar_chart_adversarial.png")  # Save just in case we want to embed this later
plt.show()

# --- Now let’s also do a pie chart ---
plt.figure(figsize=(6, 6))  # Made it square to look better for a pie
labels = ['Unchanged', 'Flipped']
sizes = [num_unchanged, num_flipped]  # Yep, repeating values here for clarity
pie_colors = ['lightgreen', 'lightcoral']

# Exploding the pie might help emphasize differences, but skipping it for now
plt.pie(sizes,
        labels=labels,
        colors=pie_colors,
        autopct='%1.1f%%',  # Format as percentage
        startangle=90,  # Rotate to make it feel more symmetrical
        shadow=True)    # Just adds a little depth
plt.title('Prediction Stability Under Adversarial Attacks')
plt.savefig("pie_chart_adversarial.png")  # Again, saving in case this needs to be included elsewhere
plt.show()
