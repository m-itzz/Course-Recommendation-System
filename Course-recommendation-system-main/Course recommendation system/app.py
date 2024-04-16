#core pkg
import streamlit as st
import streamlit.components.v1 as stc

# load EDA
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

#loading dataset
def load_data(data):
    df = pd.read_csv(data)
    return df

# Vectorize + Cosine Similarity Matrix

def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)
    # getting Cosine
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat

#Recommendation System
@st.cache_data
def get_recommendation(title,cosine_sim_mat,df,num_of_rec=5):
    #indices of the course
    course_indices = pd.Series(df.index, index=df['course_title']).drop_duplicates()
    # index of course 
    idx = course_indices[title]

    #look into the cosine matr for that index
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True) 
    selected_course_indices = [i[0] for i in sim_scores[1:]]
    selected_course_scores = [i[0] for i in sim_scores[1:]]

    #get dataframe and title
    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = selected_course_scores
    final_recommend_courses = result_df[['course_title', 'similarity_score','url','price','num_subscribers']]
    return final_recommend_courses

# CSS style
RESULT_TEMP = """
<div style="width:800px;height:100%;margin:3px;padding:5px;position:relative;border-radius:5px;
; background-color: rgb(38, 39, 48);
  border-left: 7px solid black;">
<p style="color:white;"><span style="color:white; font-family:Source Sans Pro;font-size:30px;font-weight=bold">{}</span></p>
<p style="color:white;"><span style="color:white; font-family:Helvetica">Similarity Score: </span>{}</p>
<p style="color:white;"><span style="color:white; font-family:Helvetica">Link: </span>{}</p>
<p style="color:white;"><span style="color:white; font-family:Helvetica">Price: </span>{}</p>
<p style="color:white;"><span style="color:white; font-family:Helvetica">Students:</span>{}</p>
</div>
"""

@st.cache_data
def search_term_if_not_found(term,df):
    result_df = df[df['course_title'].str.contains(term)]
    return result_df

def main():

    st.title("Course Recommendation App")

    menu = ["Home", "Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)


    df =load_data("data/udemy_courses.csv")
    if choice == "Home":
        st.subheader("Home")
        st.dataframe(df.head(10))


    elif choice == "Recommend":
        st.subheader("Recommend Courses")
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
        search_term = st.text_input("Search")
        num_of_rec = st.sidebar.number_input("Numebr",4,30,7)
        if st.button("Recommend"):
            if search_term is not None:
                try:
                     results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)
                     with st.expander("Result as JSON"):
                         results_json = results.to_dict('index')
                         st.write(results_json)
                     for row in results.iterrows():
                            rec_title = row[1][0]
                            rec_score = row[1][1]
                            rec_url = row[1][2]
                            rec_price = row[1][3]
                            rec_num_sub = row[1][4]

                            stc.html(RESULT_TEMP.format(rec_title,rec_score,rec_url,rec_price,rec_num_sub),height=350)
                            
                except:
                    results = "Not Found"
                    st.warning(results)
                    st.info("Suggested Options include")
                    result_df = search_term_if_not_found(search_term,df)
                    st.dataframe(result_df)

    else:
        st.subheader("About")
        st.text("Built wit Streamlit and Pandas")
if __name__ == '__main__':
    main()