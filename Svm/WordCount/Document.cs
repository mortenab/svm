using System.Collections.Generic;
using System.Linq;

namespace Svm.WordCount {
  /// <summary>
  /// Represents a document where the unique words have been counted.
  /// The words are sorted by their ID. 
  /// </summary>
  public class Document {
    public Document( IEnumerable<WordFrequency> words ) {
      Words = words.OrderBy( w => w.WordId ).ToList();
    }

    /// <summary>
    /// Gets the words occurrences ordered by id.
    /// </summary>
    public IReadOnlyList<WordFrequency> Words { get; }
  }
}
