namespace Svm.WordCount {
  /// <summary>
  /// The frequency of a specific word.
  /// </summary>
  public class WordFrequency {
    public WordFrequency( int wordId, int frequency ) {
      WordId = wordId;
      Frequency = frequency;
    }

    /// <summary>
    /// The ID of the word.
    /// </summary>
    public int WordId { get; set; }
    
    /// <summary>
    /// The frequency of the word.
    /// </summary>
    public int Frequency { get; set; }
  }
}
