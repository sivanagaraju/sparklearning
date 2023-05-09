from __future__ import print_function 
import sys 
from pyspark.sql import SparkSession 

#<1> Import the print() function
#<2> Import System-specific parameters and functions
#<3> Import SparkSession from the pyspark.sql module
#<4> Make sure that we have 2 parameters in the command line
#<5> Create an instance of a SparkSession object by using the builder pattern SparkSession.builder class
#<6> Define input path (this can be a file or a directory containing any number of files
#<7> Read input and create the first RDD as RDD[String] where each object has this foramt: "key,number"
#<8> Create (key, value) pairs RDD as (key, number)
#<9> Use combineByKey() to create (key, (sum, count)) per key
#<10> Apply the mapValues() transformation to find final average per key

#===================
# function:  create_pair() to accept 
# a String object as "key,number" and  
# returns a (key, number) pair.
#
# record as String of "key,number"
def create_pair(record):
    tokens = record.split(",")
    # key -> tokens[0] as String
    # number -> tokens[1] as Integer
    return (tokens[0], int(tokens[1]))
# end-of-function
#===================
# function:  `add_pairs` accept two
# tuples of (sum1, count1) and (sum2, count2) 
# and returns sum of tuples (sum1+sum2, count1+count2).
#
# a = (sum1, count1)
# b = (sum2, count2)
def add_pairs(a, b):
    # sum = sum1+sum2
    sum = a[0] + b[0]
    # count = count1+count2 
    count = a[1] + b[1]
    return (sum, count)
# end-of-function
#===================
def main():
    # <4>    
    if len(sys.argv) != 2:  
        print("Usage: ", __file__, " <input-path>", file=sys.stderr)
        exit(-1)
    #end-if
    
    # <5>
    spark = SparkSession.builder.getOrCreate()

    #  sys.argv[0] is the name of the script.
    #  sys.argv[1] is the first parameter
    # <6>
    input_path = sys.argv[1]  
    print("input_path: {}".format(input_path))

    # read input and create an RDD<String>
    # <7>
    records = spark.sparkContext.textFile(input_path) 
    print("records.count(): ", records.count())
    print("records.collect(): ", records.collect())

    # create a pair of (key, number) for "key,number"
    # <8>
    pairs = records.map(create_pair)
    print("pairs.count(): ", pairs.count())
    print("pairs.collect(): ", pairs.collect())

    #============================================================
    # combineByKey(
    #   createCombiner, 
    #   mergeValue, 
    #   mergeCombiners, 
    #   numPartitions=None, 
    #   partitionFunc=<function portable_hash>
    # )
    #
    # Generic function to combine the elements for each key using 
    # a custom set of aggregation functions.
    # Turns an RDD[(K, V)] into a result of type RDD[(K, C)], 
    # for a "combined type" C.
    #
    # Users provide three functions:
    # 1. createCombiner, which turns a V into a C (e.g., 
    #    creates a one-element list)
    #    V --> C
    #
    # 2. mergeValue, to merge a V into a C (e.g., adds it 
    #    to the end of a list)
    #    V, C --> V
    #
    # 3. mergeCombiners, to combine two C's into a single one 
    #    (e.g., merges the lists)
    #    C, C --> C
    #
    # To avoid memory allocation, both mergeValue and mergeCombiners 
    # are allowed to modify and return their first argument instead 
    # of creating a new C.
    #
    # In addition, users can control the partitioning of the output RDD.
    #
    # Note V and C can be different - for example, in our example here,
    # V is an Int type (as a number) and C is  (Int, Int) as (sum, count)
    # of numbers.

    # aggregate the (sum, count) of each unique key
    # <9>
    sum_count = pairs.combineByKey(\
        lambda v : (v, 1),\
        lambda C, v: (C[0]+v, C[1]+1),\
        lambda C1,C2: (C1[0]+C2[0], C1[1]+C2[1])\
    )
    #
    print("sum_count.count(): ", sum_count.count())
    print("sum_count.collect(): ", sum_count.collect())

    # create the final RDD as RDD[key, average]
    # <10>
    # v = (v[0], v[1]) = (sum, count)
    averages =  sum_count.mapValues(lambda v : float(v[0]) / float(v[1])) 
    print("averages.count(): ", averages.count())
    print("averages.collect(): ", averages.collect())

    # done!
    spark.stop()
#end-def


if __name__ == '__main__':
    main()