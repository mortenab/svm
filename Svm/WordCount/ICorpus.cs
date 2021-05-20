namespace Svm.WordCount {
  /// <summary>
  /// A test corpus of documents. 
  /// </summary>
  public interface ICorpus {
    /// <summary>
    /// The overall frequency in all documents.
    /// </summary>
    int GetFrequency( int wordId );
    
    /// <summary>
    /// The number of documents in the corpus.
    /// </summary>
    int CorpusSize { get; }
  }
}
