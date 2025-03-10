import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.markdown("<h1 style='text-align: center;'>Predicting Education & Career Success</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: orange;'>Can we accurately predict a person's starting salary based on their educational background, skills, and career-related experiences?</h3>", unsafe_allow_html=True)

st.sidebar.title("Contents")

df=pd.read_csv("education.csv") 

app_mode=st.sidebar.selectbox("Pages",["01 Business Case and Data","02 Data Visualization","03 Model Predictions","04 Conclusion"])
if app_mode=="01 Business Case and Data":
    st.image("ivy.jpg")
    st.markdown("<h3 style='font-size: 28px;'>Objectives</h3>", unsafe_allow_html=True)

    st.write("Our goal is to analyze the factors influencing salary and build a predictive model to estimate starting salaries based on education, skills, and experience. The dataset includes key variables such as university ranking, field of study, internships, certifications, networking, and job offers—allowing us to examine how these elements shape earning potential. As international students, we often face challenges like cultural differences, limited professional networks, and visa restrictions, which can impact our career outcomes. By leveraging this data, our model aims to provide clarity on salary determinants and offer tailored career insights to help navigate these challenges effectively.")
    st.markdown("<h3 style='font-size: 19px;'>1️⃣ Identify Key Salary Drivers</h3>", unsafe_allow_html=True)
    st.write("• Analyze correlations between salary and factors such as university ranking, GPA, field of study, internships, certifications, and networking.")
    st.write("• Perform feature importance analysis to determine which variables have the greatest impact on salary predictions.")
    st.write("• Investigate demographic factors like age and gender to uncover any disparities or trends that may impact salaries for candidates.")

    st.markdown("<h3 style='font-size: 19px;'>2️⃣ Develop a Predictive Model</h3>", unsafe_allow_html=True)
    st.write("• Use regression models (e.g., linear regression, decision trees, or random forests) to predict starting salary for students based on the factors identified.")
    
    st.markdown("<h3 style='font-size: 19px;'>3️⃣ Provide Career Insights & Recommendations</h3>", unsafe_allow_html=True)
    st.write("Generate insights such as:")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;→ Which fields of study offer the highest-paying jobs?")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;→ Does networking, or soft skills have a greater impact on salary")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;than academic performance (e.g., GPA)?")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;→ Is the number of internships or certifications more valuable in improving")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;salary potential, especially when entering competitive job markets?")
    st.write("• Develop actionable career guidance for students, helping them enhance their career prospects by identifying key strategies, such as leveraging networking opportunities, choosing the right field of study, or gaining specific certifications that improve earning potential.")

    st.markdown("### Exploring the Dataset")
    df=pd.read_csv("education.csv")
    
    st.write("A preview of the dataset is shown below:")
    st.dataframe(df.head(5))
    st.write("Source: https://www.kaggle.com/datasets/adilshamim8/education-and-career-success/data")

   
    st.markdown("### Statistics Summary:")
    st.dataframe(df.describe())
    st.markdown("<h3 style='font-size: 19px;'>Key Insights</h3>", unsafe_allow_html=True)
    st.write("• Academics: Higher university ranking and SAT scores correlate with better salaries, but internships matter more.")
    st.write("• Internships: More internships = higher job offers & starting salaries. Zero internships lead to fewer opportunities.")
    st.write("• Networking & Soft Skills: Stronger networking leads to more job offers, even with average academics.")
    st.write("• Salaries: Average starting salary is 50,563, but varies widely (~$14,495 deviation). More offers = higher pay.")
    st.write("• Career Growth: Promotions take ~3 years. Certifications and projects speed up advancement.")


elif app_mode=="02 Data Visualization":
    df=pd.read_csv("education.csv")
    st.image("money.jpg")
    st.markdown("<h3 style='font-size: 28px;'>Data Visualization</h3>", unsafe_allow_html=True)
    st.write("Now we will visualize the data in order to better understand what variables correlate with a student's starting salary.")

    list_of_var=df.columns

    st.title("Student Body by Gender")

    gender = df['Gender'].dropna()
    gender_count = gender.value_counts()
    fig, ax = plt.subplots()
    ax.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    st.title("Correlation between Soft Skills, Networking, and Starting Salary")

    feature = st.selectbox("Select a qualitative factor to analyze:", ["Networking_Score", 'Internships_Completed', 'Soft_Skills_Score', 'Job_Offers', 'Gender'])

# Plot the correlation
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=df[feature], y=df["Starting_Salary"], ax=ax)
    plt.xlabel(feature.replace("_", " "))
    plt.ylabel("Starting Salary")
    st.pyplot(fig)

    major_counts = df["Field_of_Study"].value_counts()
    st.title("Correlation Heatmap")

    df_2 = df[['University_Ranking', 'University_GPA', 'Internships_Completed', 'Soft_Skills_Score', 'Job_Offers', 'Starting_Salary']]
    df_corr = df_2.corr()


    list_columns = df_2.columns
    selected_variables = st.multiselect("Select some variables",list_columns,default=['Starting_Salary', 'University_GPA', 'Internships_Completed'])

    st.write(selected_variables)
    fig2, ax=plt.subplots(figsize=(14,8))
    df_3 = df_2[selected_variables].corr()
    sns.heatmap(df_3,annot=True,cmap="coolwarm")
    st.pyplot(fig2)

    
    st.title("University GPA vs. Starting Salary")
    df["Field_of_Study2"] = df["Field_of_Study"]
    df["Field_of_Study2"] = df["Field_of_Study2"].replace({"Mathematics": "STEM"})
    df["Field_of_Study2"] = df["Field_of_Study2"].replace({"Medicine": "STEM"})
    df["Field_of_Study2"] = df["Field_of_Study2"].replace({"Computer Science": "STEM"})
    df["Field_of_Study2"] = df["Field_of_Study2"].replace({"Engineering": "STEM"})
    df["Field_of_Study2"] = df["Field_of_Study2"].replace({"Arts": "Humanities"})
    df["Field_of_Study2"] = df["Field_of_Study2"].replace({"Business": "Business"})
    df["Field_of_Study2"] = df["Field_of_Study2"].replace({"Law": "Business"})

# Create scatter plot
    fig, ax = plt.subplots()
    sns.scatterplot(x='University_GPA', y='Starting_Salary', data=df, hue="Field_of_Study2", ax=ax)
# Customize plot
    ax.set_title('University GPA vs. Salary')
    ax.set_xlabel("University GPA")
    ax.set_ylabel("Salary")
# Show plot in Streamlit
    st.pyplot(fig)

    st.title("Average Job Offers Based on Internships Completed")


    # Group by number of internships completed and calculate the average number of job offers
    internship_job_offer_avg = df.groupby("Internships_Completed")["Job_Offers"].mean()

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(internship_job_offer_avg.index, internship_job_offer_avg.values)
    
    # Labels and title
    ax.set_xlabel("Number of Internships Completed")
    ax.set_ylabel("Average Number of Job Offers")
    ax.set_title("Average Job Offers Based on Internships Completed")

    # Show the chart in Streamlit
    st.pyplot(fig)

elif app_mode=="03 Model Predictions":
    st.image("working.jpg")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    df=pd.read_csv("education.csv")
    st.markdown("<h3 style='font-size: 28px;'>Linear Regression Model</h3>", unsafe_allow_html=True)
    st.write("Now we will create a prediction model in order to help predict the actual correlation between starting salary and the other factors. We predict university ranking, which we have observed in the previous page to have the highest correlation, will be the strongest, but this model will help us define that.")

    import streamlit as st
    import streamlit as st

# Initialize session state for navigation
    if "page" not in st.session_state:
        st.session_state["page"] = "Dataset and Features"

# Define pages
    pages = ["Dataset and Features", "Results", "Model Performance Metrics", "Predictions vs Actual"]

# Custom CSS for a full-width horizontal navigation bar
    st.markdown("""
        <style>
        .menu-container {
        display: flex;
        justify-content: space-evenly;
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .menu-item {
        flex: 1;
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        padding: 12px;
        cursor: pointer;
        border-radius: 5px;
        transition: background-color 0.3s, color 0.3s;
        text-decoration: none;
        color: black;
    }
    .menu-item:hover {
        background-color: #ddd;
    }
    .active {
        background-color: #007BFF !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Create a clickable menu bar using Streamlit buttons
    menu_html = '<div class="menu-container">'
    for page in pages:
        active_class = "active" if st.session_state["page"] == page else ""
        if st.button(page, key=page, help=f"Go to {page}", use_container_width=True):
            st.session_state["page"] = page
    menu_html += '</div>'

    st.markdown(menu_html, unsafe_allow_html=True)

# Display selected page content
    page = st.session_state["page"]

    if page == "Dataset and Features":
        st.header("📊 Dataset and Features")
        st.write("We use the following features to predict **Starting Salary**:")
        st.write("- **University_Ranking**: How well the university is ranked")
        st.write("- **University_GPA**: Student's academic performance")
        st.write("- **Internships_Completed**: Number of internships done")
        st.write("- **Soft_Skills_Score**: Communication, leadership, teamwork rating")
        st.write("- **Job_Offers**: Number of job offers received")

    elif page == "Results":
        st.header("📈 Results")
        st.write("Here are the results of the model analysis.")
        y = df['Starting_Salary']
        X = df[['University_Ranking', 'University_GPA', 'Internships_Completed', 'Soft_Skills_Score', 'Job_Offers']]
        X.head()
        X.shape
        st.write("Full dataset (5,000 rows, 5 features)")
        y.head()
        y.shape
        st.write("Target variable (y) – 5,000 salaries")

        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
        X_train.shape
        st.write("Training set (X_train) – 4,000 rows, 5 features")
        X_test.shape
        st.write("Test set (X_test) – 1,000 rows, 5 features")
        y_train.shape
        st.write("Training target (y_train) – 4,000 salaries")
        y_test.shape
        st.write("Test target (y_test) – 1,000 salaries")


        plt.figure(figsize=(10,7))
        plt.title("Actual vs. predicted house prices",fontsize=25)
        plt.xlabel("Actual test set house prices",fontsize=18)
        plt.ylabel("Predicted house prices", fontsize=18)
        plt.scatter(x="Starting_Salary", y="University_Ranking")
    
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        prediction = lr.predict(X_test)
        X_test.head(1)

        prediction[0]
        st.write("Our model's salary predictions are off by about $50,745 on average compared to the actual salaries.")
        prediction.shape
        st.write("Prediction shape – 1,000 salaries predicted")

        st.subheader("Training and Testing Data")
        st.write(f"**Training Set Size:** {X_train.shape[0]} samples")
        st.write(f"**Testing Set Size:** {X_test.shape[0]} samples")

    elif page == "Model Performance Metrics":
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        y = df['Starting_Salary']
        X = df[['University_Ranking', 'University_GPA', 'Internships_Completed', 'Soft_Skills_Score', 'Job_Offers']]
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        prediction = lr.predict(X_test)
        st.header("📉 Model Performance Metrics")
        from sklearn import metrics
        print("Mean Absolute Error :",metrics.mean_absolute_error(y_test,prediction))
        df["Starting_Salary"].max()


        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # Evaluate model performance
        mae = mean_absolute_error(y_test, prediction)
        r2 = r2_score(y_test, prediction)  # ✅ Fix for r2 score

        st.write(f"✅ Mean Absolute Error (MAE): {mae}")
        st.write(f"✅ R² Score: {r2} (Closer to 1 is better)")  # Display R² score properly
        st.write("### MAE Interpretation 📊")
        st.write("On average, our model’s predictions are off by $11,718.56 from the true salary values.")
        st.write("### R² Score Interpretation 📊")
        st.write("The R² (R-squared) score tells us how much of the variance in salaries our model can explain.")
        st.write("- **An R² of 1** means a perfect model (100% of salary variance explained).")
        st.write(f"- **Our R² = {metrics.r2_score(y_test, prediction):.4f}** → Our model explains almost **0%** of salary variation, meaning it’s **not a useful predictor** in its current form.")

    elif page == "Predictions vs Actual":
        st.header("🔍 Predictions vs Actual")
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        y = df['Starting_Salary']
        X = df[['University_Ranking', 'University_GPA', 'Internships_Completed', 'Soft_Skills_Score', 'Job_Offers']]
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        prediction = lr.predict(X_test)
        from sklearn import metrics
        print("Mean Absolute Error :",metrics.mean_absolute_error(y_test,prediction))
        df["Starting_Salary"].max()

    
# Define variables
        y = df['Starting_Salary']  # Target variable
        X = df[['University_Ranking', 'University_GPA', 'Internships_Completed', 'Soft_Skills_Score', 'Job_Offers']]  # Features

    # **2. Split Data into Training and Testing Sets**
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # **3. Train the Linear Regression Model**
        lr = LinearRegression()
        lr.fit(X_train, y_train)

    # **4. Make Predictions on Test Data**
        prediction = lr.predict(X_test)

    # **5. Model Performance Metrics**
        mae = mean_absolute_error(y_test, prediction)
        mse = mean_squared_error(y_test, prediction)
        r2 = r2_score(y_test, prediction)  # Fix: Ensure correct calculation

    # **6. Visualization: Actual vs. Predicted**
        st.write("This plot compares the actual salaries from the dataset with the predicted salaries from our model.")

        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(y_test, prediction, color="blue", label="Predicted vs Actual")
        ax.plot(y_test, y_test, color="red", linestyle="dashed", label="Perfect Fit")
        ax.set_xlabel("Actual Starting Salary")
        ax.set_ylabel("Predicted Salary")
        ax.legend()
        st.pyplot(fig)

    # **7. Display Maximum Salary**
        max_salary = df["Starting_Salary"].max()
        st.write(f"🎯 **Maximum Starting Salary in Dataset:** {max_salary}")
        st.subheader("What does this show?")
        st.write("The red diagonal line represents where the predictions should be if the model were perfect.")
        st.write("Instead of varying with actual salaries, the predictions are mostly flat, meaning the model may not be capturing the right relationships.")

elif app_mode=="04 Conclusion":
    
    st.image("grad.jpg")
    st.title('🔑 Key Relationships in the Data')

    st.subheader('High School GPA & University GPA')
    st.write("Weak positive correlation (0.0049), indicating that high school performance is not a strong predictor of university GPA.")

    st.subheader('SAT Score & Job Offers')
    st.write("Weak positive correlation (0.019), meaning SAT scores have little to no effect on job offers.")

    st.subheader('Internships Completed & Job Offers')
    st.write("A slightly negative correlation (-0.022) which is counterintuitive—it might suggest other factors like networking or field of study are more decisive.")

    st.subheader('Certifications & Salary')
    st.write("Slightly positive correlation (0.017), meaning certifications have a small impact on starting salary.")

    st.subheader('Soft Skills & Networking')
    st.write("Very weak correlation (0.001), implying that these are measured separately in this dataset.")

    st.title('🤔 Unexpected Findings')

    st.subheader('University GPA & Salary')
    st.write("Almost no correlation (0.001), suggesting GPA does not strongly influence starting salary.")

    st.subheader('University Ranking & Salary')
    st.write("Weak positive correlation (0.021), meaning attending a highly-ranked university has a small effect on earnings.")

    st.subheader('Job Offers & Salary')
    st.write("Low correlation, implying that more job offers don’t necessarily lead to higher salaries (quality over quantity?).")

    st.title('👩‍🏫 Implications for our Presentation')
    st.subheader('Quality > Quantity')
    st.write("Since the goal of our investigation is predicting career success, and our model struggles to find strong predictors, this suggests that personal factors, networking, and skills may matter more.")
    st.write("The weak correlation between university ranking and salary suggests that prestige may not guarantee higher pay.")
    st.write("The negative correlation between internships and job offers is counterintuitive and worth discussing—perhaps quality of experience matters more than quantity.")

    st.subheader('The job market nowadays:')
    st.image("study.png")

    st.subheader("Maybe it's time to question if we're even on the right path to success...💸")



    




