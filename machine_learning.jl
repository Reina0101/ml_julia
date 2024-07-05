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