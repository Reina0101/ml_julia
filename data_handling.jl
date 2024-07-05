# Part 1: Handling & using data

using LinearAlgebra

A = [
    [1 2 3]
    [4 5 6]
    [7 8 9]
]
B = [
    [7 8 9]
    [4 5 6]
    [1 2 3]
]

# Showing results
println("Matrix multiplication: ", A*B)

println("Determinant: ",det(A))
println("Trace: ",tr(A))
println("Inverse: ", inv(A))


# initializing dataframe:
using DataFrames, CSV
df = DataFrame(name=["Julia", "Robert", "Bob","Mary"], age=[12,15,45,32])
println("Dataframe: ", df)

# Selecting rows/columns similar to Pandas
# df[<rows>,<columns>]
println(df[age,])
println(df[,name])

# To get a subset out of the dataframe
subs = df[1:3,"age"]
println(subs)

# Fist entry is not 0 but 1 
subs_2 = df[1:3,:]
println(subs_2)

# To get a single column, access it using . (similar to Pandas)
names = df.name 
println(names)

# For queries, apply filters directly on to the dataframe
older = df[df.age .>15,:]
println("People older than 15", older)

# sorting data can be done by invoking sort()
println("Dataframe before sorting:", df)
df_s = sort(df,"age")
println("Dataframe after sort", df_s)
df_rs = sort(df,"age",rev=true)
println("Dataframe reverse soft", df_rs)

# Adding additional columns:
df.sex = ["female","male","???","female"]

# Query using SQL-esque statements
new_df = select(df,Not("sex"))

# Grouping
group_df = groupby(df,"sex")
combine(group_df,nrow => "count")

combine(group_df, 
	nrow => "count", 
	"age" => ((rows) -> sum(rows)/length(rows)) => "Average Age"
)
