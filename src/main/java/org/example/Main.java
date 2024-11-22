package org.example;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.lucene.analysis.ru.RussianLightStemmer;
import scala.Tuple2;

import java.io.Serializable;
import java.util.*;

/**
 * Lab2_Potapkin
 *
 * @author Danila Potapkin
 * @since 18.11.2024
 */


public class Main {
	/**
	 * 10. Николай Васильевич Гоголь – Нос
	 */
	public static void main(String[] args) {
		// Настройка Spark
		SparkConf conf = new SparkConf().setAppName("GogolNoseAnalysis").setMaster("local[*]");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// Чтение текста
		String filePath = "D:\\МАГИСТРАТУРА\\2 КУРС\\3 семестр\\Большие данные\\Lab2_bigData_Potapkin\\gogol_nose.txt";
		JavaRDD<String> lines = sc.textFile(filePath);

		// Очистка текста
		JavaRDD<String> cleanedWords = lines
			.flatMap((FlatMapFunction<String, String>) line -> Arrays.asList(line.toLowerCase()
				.replaceAll("[^a-zа-яё]", " ")
				.split("\\s+")).iterator())
			.filter((Function<String, Boolean>) word -> !StopWords.contains(word));

		// Подсчет слов
		JavaPairRDD<String, Integer> wordCounts = cleanedWords
			.mapToPair((PairFunction<String, String, Integer>) word -> new Tuple2<>(word, 1))
			.reduceByKey(Integer::sum);

		long totalWordCount = wordCounts
			.map(Tuple2::_2)
			.reduce(Integer::sum);

		// Топ-50 самых частых слов
		List<Tuple2<String, Integer>> top50Most = wordCounts
			.takeOrdered(50, new WordCountComparator(false));

		// Топ-50 наименее частых слов
		List<Tuple2<String, Integer>> top50Least = wordCounts
			.takeOrdered(50, new WordCountComparator(true));

		// Вывод результатов
		System.out.println("Количество слов в тексте:" + totalWordCount);

		System.out.println("Топ-50 наиболее частых слов:");
		top50Most.forEach(System.out::println);

		System.out.println("\nТоп-50 наименее частых слов:");
		top50Least.forEach(System.out::println);

		// Выполнение стемминга
		JavaRDD<String> stemmedWords = cleanedWords.map((Function<String, String>) Stemmer::stem);

		// Подсчет слов после стемминга
		JavaPairRDD<String, Integer> stemmedWordCounts = stemmedWords
			.mapToPair((PairFunction<String, String, Integer>) word -> new Tuple2<>(word, 1))
			.reduceByKey(Integer::sum);

		// Топ-50 после стемминга
		List<Tuple2<String, Integer>> stemmedTop50Most = stemmedWordCounts
			.takeOrdered(50, new WordCountComparator(false));

		List<Tuple2<String, Integer>> stemmedTop50Least = stemmedWordCounts
			.takeOrdered(50, new WordCountComparator(true));

		System.out.println("\nТоп-50 наиболее частых слов после стемминга:");
		stemmedTop50Most.forEach(System.out::println);

		System.out.println("\nТоп-50 наименее частых слов после стемминга:");
		stemmedTop50Least.forEach(System.out::println);

		sc.close();
	}

	// Стоп-слова
	private static final Set<String> StopWords = new HashSet<>(Arrays.asList("и", "в", "на", "с", "по", "из", "у", "что", "это", "как", "а"));

	// Компаратор для сортировки
	private static class WordCountComparator implements Comparator<Tuple2<String, Integer>>, Serializable {
		private final boolean ascending;

		public WordCountComparator(boolean ascending) {
			this.ascending = ascending;
		}

		@Override
		public int compare(Tuple2<String, Integer> o1, Tuple2<String, Integer> o2) {
			return ascending ? o1._2.compareTo(o2._2) : o2._2.compareTo(o1._2);
		}
	}

	private static class Stemmer {
		private static final RussianLightStemmer stemmer = new RussianLightStemmer();

		public static String stem(String word) {
			try {
				char[] chars = word.toCharArray();
				int length = stemmer.stem(chars, chars.length);
				return new String(chars, 0, length);
			} catch (Exception e) {
				System.err.println("Ошибка стемминга для слова: " + word);
			}
			return word;
		}
	}
}
