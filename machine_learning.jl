# Part 3: machine learning

# Add packages
# using Pkg
# Pkg.add("DataFrames")
# Pkg.add("CSV")

# Import modules
using DataFrames, CSV

# Load training data to data frame
train_df = CSV.read("data/train.csv", DataFrame)
println(train_df)

# To see the summary information, use describe()
println(describe(train_df))

# Making a few small edits to the data, since some contain missing information
train_df = dropmissing(train_df,"Embarked")
# This will remove all rows with missing values in the Embarked column

# The median age of the people is 28, let's use this value to fill in the gaps in the data 
train_df.Age = replace(train_df.Age,missing=>28)
println("Everyone has an age: ", train_df.Age)

# The Cabin column has a lot of missing data, let's not take this into the machine learning
train_df = select(train_df, Not("Cabin"))
# Cabin is no longer in the dataset

# Now to change certain columns to numerical 
train_df = select(train_df,Not(["PassengerId","Name"]))
combine(groupby(train_df,"Embarked"),nrow=>"count")

# To keep the data integrity, let's convert the labels to numbers
train_df.Embarked = Int64.(replace(train_df.Embarked, "S" => 1, "C" => 2, "Q" => 3))

# See what the sex distribution is in our dataset
combine(groupby(train_df,"Sex"),nrow=>"count")

# And convert this to numbers, Julia uses 1 as first so let's keep this quirk 
train_df.Sex = Int64.(replace(train_df.Sex, "female" => 1, "male" => 2))

# As shown below, there are a lot of ticket categories, this is too much less relevant information
combine(groupby(train_df,"Ticket"),nrow=>"count")

# Let's drop the information
train_df = select(train_df,Not("Ticket"))

# Now the data looks a lot better! 
println(describe(train_df))

# part3.5
# Let's get a feel for the data
using Plots
# Group dataset by "Survived" column
survived = combine(groupby(train_df,"Survived"), nrow => "Count")
# Display the data on bar chart
plot(survived.Survived, survived.Count, title="Survived Passengers", label=nothing, seriestype="bar", texts=survived.Count)
# Modify X axis to display text labels instead of numbers
xticks!([0:1:1;],["Not Survived","Survived"])


# A lot more females have survived, there is a large skew in the data
# Group dataset by Sex column and show only rows where Survived=1
survived_by_sex = combine(groupby(train_df[train_df.Survived .== 1,:],"Sex"), nrow => "Count")
# Display the data on bar chart 
plot(survived_by_sex.Sex, survived_by_sex.Count, title="Survived Passengers by Sex", label=nothing, seriestype="bar", texts=survived_by_sex.Count)
# Modify X axis to display text labels instead of numbers
xticks!([1:1:2;],["Female","Male"])


#
# Group dataset by PClass column and show only rows where Survived=0
death_by_pclass = combine(groupby(train_df[train_df.Survived .== 0,:],"Pclass"), nrow => "Count")
# Display the data on bar chart 
plot(death_by_pclass.Pclass, death_by_pclass.Count, title="Dead Passengers by Ticket class", label=nothing, 
    seriestype="bar", texts=death_by_pclass.Count)
# Modify X axis to display text labels instead of numbers
xticks!([1:1:3;],["First","Second","Third"])

### Machine learning:
using DecisionTree, SciKitLearn.CrossValidation

# Using random forest
# Put "Survived" column to labels vector
y = train_df[:,"Survived"]
# Put all other columns to features matrix (important to convert to "Matrix" data type)
X = Matrix(train_df[:,Not(["Survived"])])

# Create Random Forest Classifier with 100 trees
model = RandomForestClassifier(n_trees=100)

# Train the model, using features matrix and labels vector
fit!(model,X,y)

# Evaluate the accuracy of predictions using Cross Validation
accuracy = minimum(cross_val_score(model, X, y, cv=5))
println("The accuracy is not bad at all: ", accuracy)

# Let's grab the test data set to see how it actually performed
test_df = CSV.read("data/test.csv",DataFrame)
describe(test_df)

PassengerId = test_df[:,"PassengerId"]
# Lets do the same steps for test data
# Repeat the same transformations as we did for training dataset
test_df = select(test_df,Not(["PassengerId","Name","Ticket","Cabin"]))
test_df.Age = replace(test_df.Age,missing=>28)
test_df.Embarked = replace(test_df.Embarked,"S" => 1, "C" => 2, "Q" => 3)
test_df.Embarked = convert.(Int64,test_df.Embarked)
test_df.Sex = replace(test_df.Sex,"female" => 1,"male" => 2)
test_df.Sex = convert.(Int64,test_df.Sex)

# In addition, replace missing value in 'Fare' field with median
test_df.Fare = replace(test_df.Fare,missing=>14.4542)

# Moment of truth to see how the RF-model performed
Survived = predict(model, Matrix(test_df)) 
println("Prediction of survivors: ", survived)

### Exporting model for deployment:
Pkg.add("JLD2")
using JLD2
save_object("titanic.jld2", model)

using JLD2, DecisionTree
model = load_object("titanic2.jld2")
survived = predict(model,[1 2 35 0 2 144.5 1])
println(survived)


# Returns 1 if a passenger with
# specified 'data' survived or 0 if not
function isSurvived(data)
	model = load_object("titanic2.jld2")
	survived = predict(model,data)
	return survived[1]
end
