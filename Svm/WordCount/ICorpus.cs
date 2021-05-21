namespace Svm.WordCount {
  /// <summary>
  /// A test corpus of documents. 
  /// </summary>
  public interface ICorpus {
    /// <summary>
    /// The number of documents containing the specified word.
    /// </summary>
    int GetDocumentFrequency( int wordId );
    
    /// <summary>
    /// The number of documents in the corpus.
    /// </summary>
    int CorpusSize { get; }
  }
}
