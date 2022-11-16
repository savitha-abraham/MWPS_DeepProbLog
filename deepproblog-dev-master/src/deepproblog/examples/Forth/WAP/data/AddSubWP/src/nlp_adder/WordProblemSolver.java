package nlp_adder;

import java.util.Properties;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;

public class WordProblemSolver {
	
	public static void main(String[] main) {
		Properties props = new Properties();
	    props.put("annotators", "tokenize, ssplit, pos, lemma, ner,parse,dcoref");
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		solveWordProblems("Joan spent $ 15 on shorts and $ 14.82 on a jacket , and $ 12.51 on a shirt . She went to 3 shops . In total , how much money did Joan spend on clothing ?", pipeline);
	}

	public static String solveWordProblems(String problem, StanfordCoreNLP pipeline) {
		String simplifiedProblem = Parser.parse(problem, pipeline);
		System.out.println("ha"+simplifiedProblem);
	    LinguisticInfo extractedInformation = (new SentencesAnalyzer()).extract(simplifiedProblem, pipeline);
	    KnowledgeRepresenter.clear();
	    KnowledgeRepresenter.represent(extractedInformation, simplifiedProblem);
	    KnowledgeRepresenter.solve();
	    System.err.println(KnowledgeRepresenter.finalAns);
	    return KnowledgeRepresenter.finalAns;
	}
}
