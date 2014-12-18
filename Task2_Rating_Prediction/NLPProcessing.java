import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Properties;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
/*
 * Class performs the sentiment analysis and create the features for each users.
 */

public class NLPProcessing 
{
	static BufferedReader br;
	static JSONParser parser;
	static StanfordCoreNLP pipeline;
	static Properties props;

	/*
	 * Main method which initializes the stafords nlp properties with required annotators.
	 */
	public static void main(String[] args) throws IOException, ParseException 
	{
		parser = new JSONParser();
		props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit,parse, sentiment");
		pipeline = new StanfordCoreNLP(props);

		String filePath="<FilePath>";
		//call function to read the users info and perform the analysis
		fileRead(filePath);
	}

	/*
	 * Method perfroms the sentimental analysis and creates seperate file for each user consists of features required for training the user.
	 */
	private static void fileRead(String filePath) throws IOException, ParseException 
	{
		String newLine;
		br = new BufferedReader(new FileReader(filePath));
		int count =0;

		//looping over each user
		while((newLine = br.readLine())!=null)
		{
			JSONObject jsonObject = (JSONObject)parser.parse(newLine);
			String user_id = (String) jsonObject.get("user_id");
			JSONArray reviewsArray = (JSONArray) jsonObject.get("reviews");
			String text;
			//file writers , writing training data and testing data separately..
			FileWriter fwTraining = new FileWriter("TrainingData/user"+user_id+".csv");
			FileWriter fwTestingStars = new FileWriter("TestingDataWithStars/user"+user_id+".csv");
			fwTraining.write("very_negative,negative,neutral,positive,very_positive,stars"+"\n");
			fwTestingStars.write("very_negative,negative,neutral,positive,very_positive,stars"+"\n");
			
			int size = reviewsArray.size();
			//looping over each review given by the user to extract the review text and perform the analysis
			for(int i=0;i<size;i++)
			{
				JSONObject review = (JSONObject)parser.parse(reviewsArray.get(i).toString());
				text = (String)review.get("text");
				Long stars = (Long) review.get("stars");
				//fucntion call to perfrom the sentiment alaysis given the review text.
				double[] senti =findSentiment(text);
				
				//80:20 divison for training and testing data
				if(i<size*0.8)
					fwTraining.write(senti[0] + ","+senti[1] + ","+senti[2] + ","+senti[3] + ","+senti[4] + ","+stars+"\n");
				else{
					fwTestingStars.write(senti[0] + ","+senti[1] + ","+senti[2] + ","+senti[3] + ","+senti[4] + ","+stars+"\n");
				}
			}

			fwTraining.close();
			fwTestingStars.close();
			System.out.println("count : "+ ++count);
		}
		br.close();
	}


	/*
	 * Function accepts the reviews text and perfroms sentimental analysis to return the array of size five . each cell corresponds to the sentiment analysis class and 
	 * has value of count of sentence falling into that sentiment category
	 */
	private static double[] findSentiment(String text) 
	{
		//array initialization
		double sentiArray[] = {0,0,0,0,0};			// [vneg,neg,neutral,pos,vpos]
		if (text!= null && text.length() > 0) 
		{
			Annotation annotation = pipeline.process(text);
			int sentiment=0;
			List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
			int numberOfSentences = sentences.size();
			//looping over each sentence in the text.
			for (CoreMap sentence : sentences) 
			{
				Tree tree = sentence.get(SentimentCoreAnnotations.AnnotatedTree.class);
				sentiment = RNNCoreAnnotations.getPredictedClass(tree); 
				//increasing counter of particular sentiment cell.
				sentiArray[sentiment]++;
			}
			//Normalizing the values by dividing all the array values with sentence count.
			for(int i=0;i<sentiArray.length;i++)
				sentiArray[i] /= numberOfSentences;
		}	
		return sentiArray;
	}
}