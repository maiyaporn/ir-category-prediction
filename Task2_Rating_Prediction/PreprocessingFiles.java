import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import com.mongodb.BasicDBObject;
import com.mongodb.DB;
import com.mongodb.DBCollection;
import com.mongodb.DBCursor;
import com.mongodb.DBObject;
import com.mongodb.MongoClient;
import com.mongodb.util.JSON;

/*
 * This class is used to preprocess files for using it further for the sentimental analysis.
 */
public class PreprocessingFiles
{
	static MongoClient mongo;
	static DB yelpdb;
	static DBCollection reviewCollection;
	static BufferedReader br;
	static FileWriter fw;
	/*
	 * This is main function which initiate mongo connections and connects to the collection.
	 */
	public static void main(String[] args) throws IOException
	{
		//mongoDB initialization
		mongo = new MongoClient("localhost",27017);
		yelpdb = mongo.getDB("yelpdb");
		reviewCollection = yelpdb.getCollection("yelpcollection");
		fw = new FileWriter("<FilePath>");
		String inputFilePath = "<FilePath>";
		
		//Inserts the reviews file into mongodb line by line
		insertIntoMongoDb("<filePath>");
		
		//reading users from the file which contains only users with more than 100 reviews. Users where retrieved from mongodb join operations and exported in a file to read.
		List<String> users = readUsers(inputFilePath); 
		
		//grouping user data together.
		groupMongoData(users);
	}

	/*
	 *Function reads the users from file and returns arraylist of userid
	 */
	private static List<String> readUsers(String inputFilePath) throws IOException {
		// TODO Auto-generated method stub
		List<String> users = new ArrayList<>();
		br = new BufferedReader(new FileReader(inputFilePath));
		String newLine;
		while((newLine=br.readLine())!=null)
		{
			users.add(newLine.split(",")[0]);
		}
		return users;
	}

	
	/*
	 * Function take input as list of userid and for every userid , group all the reviews and writes it into a file.
	 */
	@SuppressWarnings("unchecked")
	private static void groupMongoData(List<String> users) throws IOException
	{

		@SuppressWarnings("rawtypes")
		int count=0;

		for(String user : users)
		{
			//initiation of one user object 
			JSONObject obj = new JSONObject();
			JSONArray jsonarray = new JSONArray();
			obj.put("user_id", user);
			//adding fields to object
			BasicDBObject fields = new BasicDBObject().append("text", 4).append("review_id", 4).append("stars", 4).append("business_id", 4);
			BasicDBObject query = new BasicDBObject().append("user_id", user);
			DBCursor results = reviewCollection.find(query, fields);
			//retriveing reviews for particular users and appending all the reviews.
			while (results.hasNext())
			{
				JSONObject reviewObj = new JSONObject();
				DBObject res = results.next();
				reviewObj.put("review_id", res.get("review_id"));
				reviewObj.put("business_id", res.get("business_id"));
				reviewObj.put("text", res.get("text"));
				reviewObj.put("stars", res.get("stars"));
				jsonarray.add(reviewObj);
			}
			obj.put("reviews", jsonarray);
			//writing the user object into file
			fw.write(obj.toJSONString()+"\n");
			++count;
			//count to check on console which user is being processed.
			System.out.println("user count : "+ count);
			
		}
		fw.close();
		
	}
	
	/*
	 * Function takes the file path as parameter and read the file using buffered reader to insert the data in the mongodb.
	 */
	private static void insertIntoMongoDb(String inputFilePath) throws IOException
	{
		br = new BufferedReader(new FileReader(inputFilePath));
		String newLine;

		while((newLine=br.readLine())!=null)
		{
			//casting object into json
			DBObject dbObj = (DBObject) JSON.parse(newLine);
			//inserting the json in mongodb
			reviewCollection.insert(dbObj);
		}
	}


}
