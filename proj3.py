# DecodeLabs Project 3
# Career Recommendation System
# Batch 2026

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# job data - manually entered
jobs = [
    "Data Scientist",
    "Cloud Engineer",
    "Python Developer",
    "Security Analyst",
    "Mobile Dev",
    "Database Admin",
    "Game Developer",
    "Frontend Dev",
    "DevOps Engineer",
    "Backend Dev",
    "ML Engineer",
    "Web Developer"
]

# skills needed for each job
skills_data = [
    "python sql machine learning statistics pandas numpy",
    "aws azure docker kubernetes terraform linux cloud",
    "python django flask rest api postgresql git",
    "network security firewalls python bash incident response",
    "swift kotlin react native flutter android ios",
    "sql postgresql oracle database tuning backup recovery",
    "csharp unity 3d modeling game design cplusplus unreal",
    "javascript html css react vue angular typescript",
    "docker jenkins kubernetes linux aws terraform ansible",
    "java spring boot microservices rest api hibernate sql",
    "python tensorflow pytorch scikit-learn keras nlp",
    "javascript html css react nodejs express mongodb"
]

print("=" * 50)
print("CAREER MATCHER - tell me your skills")
print("=" * 50)

# get user input - 3 skills
user_skills = []
print("\nEnter 3 skills you have:\n")

for i in range(3):
    skill = input(f"Skill {i + 1}: ").strip().lower()
    user_skills.append(skill)
    print(f"ok got {skill}\n")

print("-" * 50)
print(f"Your skills: {user_skills[0]}, {user_skills[1]}, {user_skills[2]}")
print("-" * 50)

# combine everything for vectorization
all_text = skills_data.copy()
user_text = " ".join(user_skills)
all_text.append(user_text)

# create tf-idf vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_text)

# separate job vectors and user vector
job_vectors = tfidf_matrix[:-1]
user_vector = tfidf_matrix[-1]

# calculate similarity scores
scores = cosine_similarity(user_vector, job_vectors).flatten()

# sort jobs by score
ranked = []
for i in range(len(jobs)):
    ranked.append((jobs[i], scores[i]))

# sort manually
for i in range(len(ranked)):
    for j in range(i + 1, len(ranked)):
        if ranked[j][1] > ranked[i][1]:
            ranked[i], ranked[j] = ranked[j], ranked[i]

print("\n" + "=" * 50)
print("TOP RECOMMENDATIONS FOR YOU")
print("=" * 50)

# show top 3
for i in range(3):
    job_name = ranked[i][0]
    match_percent = ranked[i][1] * 100

    # simple bar
    bar_len = int(match_percent / 10)
    bar = ""
    for x in range(bar_len):
        bar = bar + "#"
    for x in range(10 - bar_len):
        bar = bar + "-"

    print(f"\n{i + 1}. {job_name}")
    print(f"   Match: {match_percent:.1f}% [{bar}]")

# show all results
print("\n" + "=" * 50)
print("ALL RESULTS")
print("=" * 50)

for i in range(len(ranked)):
    print(f"{i + 1:2}. {ranked[i][0]:<20} {ranked[i][1] * 100:5.1f}%")

# find what skills are missing for top job
best_job = ranked[0][0]
best_idx = jobs.index(best_job)
needed_skills = skills_data[best_idx].split()

print("\n" + "=" * 50)
print("WHAT TO LEARN NEXT")
print("=" * 50)
print(f"\nTo get into {best_job}, learn:\n")

user_skills_lower = [s.lower() for s in user_skills]
count = 0
for skill in needed_skills:
    if skill.lower() not in user_skills_lower and count < 6:
        print(f"   - {skill}")
        count = count + 1

print("\n" + "=" * 50)
print("DONE - Project 3 Complete")
print("=" * 50)


