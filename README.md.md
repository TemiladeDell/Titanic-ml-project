# **# Titanic ML Project**



##### 	**This project predicts survival on the Titanic using machine 	learning.**



###### **## Files**

\- TitanicData.ipynb

\- train.csv

\- test.csv





### **Project Rationale \& Thought Process (Titanic ML Workflow)**

#### **Foundations and Initial Assumptions**



&nbsp;	Most of my foundational understanding comes from Hands-On Machine Learning with Scikit-Learn, Keras \& TensorFlow. 	Initially, I believed machine learning was mostly about cleaning data — dropping columns, replacing missing values 	with means, and focusing primarily on numerical features.



&nbsp;	However, working with the Titanic dataset corrected that assumption. Every column carries potential information, 	and careless preprocessing can silently destroy signal. The dataset forced me to slow down and think more 	deliberately about what each feature represents, not just how convenient it is to process.



##### **Data Organization and Workflow Decisions**



&nbsp;	I placed all CSV files directly inside the project folder. Typing long absolute paths like C:\\Desktop\\... felt 	unnecessary and brittle. Keeping everything local to the project makes experimentation faster, reproducible, and 	easier to reason about.



&nbsp;	Even though this can get messy in large projects, this is a small prediction model. The trade-off favored speed 	and clarity over strict separation.



&nbsp;	I also created copies of the dataset in memory rather than modifying the original. Conceptually, this is like 	saying: “I don’t want to alter the source of truth; I want a working copy.”

&nbsp;	This aligns with how RAM is used — fast, temporary, and safe for experimentation.



##### **Early Data Inspection**



&nbsp;	Using data.head() was intentional. Seeing the first few rows is far more efficient than trying to mentally 	visualize a dataset or constantly switching tabs between files. Jupyter’s design makes the columns permanently 	visible, which reduces cognitive load.



&nbsp;	Good developer tools matter — many function names exist because they map cleanly to how humans think.



##### **Feature Elimination: Cabin and Ticket**

###### **Cabin**



&nbsp;	My initial idea was to extract the first letter of the cabin and one-hot encode it, since higher decks generally 	correlate with wealth. However, over 600 of the 800+ entries were missing.



&nbsp;	I explored imputing cabin based on fare:



&nbsp;		Higher fare → higher deck

&nbsp;	But inconsistencies emerged:



&nbsp;		Some passengers paid $70 and were in D



&nbsp;		Others paid $40 and were in C



&nbsp;	A possible workaround would involve probabilistic assignment:



&nbsp;		$35–$100 → randomized between C and D



&nbsp;		$100+ → randomized between A and B



&nbsp;	Combined with real-world assumptions (more poor than rich, workers housed lower), this could generate a synthetic 	cabin feature.



&nbsp;	However, for a beginner competition, this added complexity and assumptions that could distort learning. I chose to 	exclude Cabin to understand how far the raw data could go without heavy inference.



###### **Ticket**



&nbsp;	Ticket numbers mix letters and digits and function mostly as identifiers. While ticket prefixes could reflect 	class (similar to airline boarding passes), extracting that meaning would require deeper historical research.



&nbsp;	Since this would again move beyond the scope of a learning-focused project, I removed it.



&nbsp;	Late Realizations: PassengerId and Name



&nbsp;		Initially, I assumed the model would “brush over” PassengerId and Name. This was incorrect.



&nbsp;	Think of a baby learning language:

&nbsp;		If you speak English but randomly inject French, Spanish, Japanese, and Italian, the baby becomes 	confused. Even 	though all languages are meaningful, they introduce noise relative to the task.

&nbsp;	

###### **Name**



&nbsp;	Names could encode social status (titles, lineage), but extracting that signal requires research and careful 	feature engineering. Without that effort, names act as high-entropy noise.



###### **PassengerId**



&nbsp;	PassengerId is especially dangerous. To humans, it’s just a label. To a model, it’s an ordinal number.



&nbsp;	If higher IDs coincidentally correlate with survival, the model may falsely learn:



&nbsp;		“Higher ID → higher survival probability”



&nbsp;		This mirrors how race placements work versus numeric magnitude:



&nbsp;		1st place beats 4th place in real life



&nbsp;		But mathematically, 4 > 1



&nbsp;	This mismatch makes PassengerId misleading and harmful.



###### **Categorical Encoding Decisions**

&nbsp;		Pclass



&nbsp;		Even though first class is socially superior, numerically:



&nbsp;			3 > 1



&nbsp;		To avoid the model learning false numeric relationships, I one-hot encoded Pclass.



###### **Sex and Embarked**



&nbsp;	Gender has no inherent numeric order, yet survival rates differ historically. One-hot encoding preserves this 	distinction without implying magnitude.



&nbsp;	Embarked is more subtle. The port of embarkation reflects geographic and social class separation — rich, middle-	class, and poor passengers often boarded from different locations. Encoding this feature preserves meaningful 	socio-economic structure.



##### **Defining X and Y**



&nbsp;	Following the equation of a line (y = mx + c):



&nbsp;		X represents inputs (features)



&nbsp;		Y represents outputs (labels)



&nbsp;	For this problem:



&nbsp;		Y = Survived (1 = lived, 0 = died)



##### **Model Selection Logic**

###### **Logistic Regression**



&nbsp;	I started with Logistic Regression because many relationships appeared roughly linear:



&nbsp;		Wealth ↑ → Survival ↑



&nbsp;		Class ↑ → Survival ↑



&nbsp;	Logistic Regression uses a sigmoid function, mapping linear combinations into probabilities between 0 and 1. It’s 	still fundamentally linear — just constrained to probabilistic output.



&nbsp;	This made it a strong baseline.



###### **Decision Tree \& Random Forest**



&nbsp;	However, real-world relationships are rarely purely linear.



&nbsp;	Some interactions behave more like polynomial relationships:



&nbsp;	A feature may help at one range and harm at another



&nbsp;	Effects may flip sign entirely



&nbsp;	Decision Trees and Random Forests capture these non-linear interactions. Adjusting tree depth and number of 	estimators is like pairing foods:



&nbsp;		Pizza + Pepsi works



&nbsp;		Pizza + Porridge doesn’t



&nbsp;	The goal is to find combinations that generalize well.



###### **Train/Test Splitting and Stratification**



&nbsp;	Splitting data randomly can distort population structure.



&nbsp;	For example:



&nbsp;		Real population: 55% women, 45% men



&nbsp;		Sample: 50% women, 50% men



&nbsp;	That mismatch skews learning.



&nbsp;	Stratified splitting ensures both training and test sets reflect the true distribution, allowing the model to 	generalize more reliably.



##### **Results and Reality Check**



&nbsp;	Voting Classifier validation accuracy: 83.7%



&nbsp;	Submission score: ~75%



&nbsp;	Logistic Regression validated lower but submitted higher



&nbsp;	This reversal happens. Validation is an estimate, not truth. Models behave differently when exposed to unseen 	distributions.



&nbsp;	Learning a model is like raising a child — you guide, test, correct, and accept imperfection.



##### **Final Reflection**



This project taught me that machine learning is not about blindly cleaning data or chasing accuracy. It’s about:



Understanding what features mean



Knowing when assumptions help or hurt



Accepting that uncertainty is part of learning



This notebook reflects how I think — not just what I coded.



