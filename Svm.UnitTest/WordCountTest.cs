using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using Svm.WordCount;

namespace Svm.UnitTest {
  public class WordCountTest {
    private static readonly List<(string doc, string label)> Documents = new() {
      ( "Blah blah blah", "class1" ),
      ( "Blah blah blah", "class1" ),
      ( "Blah blah blah", "class1" ),
      ( "Blah blah blah", "class1" ),
      ( "Blah blah blah", "class1" ),
      ( "Blah blah blah", "class1" ),
      ( "Yada yada yada", "class2" ),
      ( "Yada yada yada", "class2" ),
      ( "Yada yada yada", "class2" ),
      ( "Yada yada yada", "class2" ),
      ( "Yada yada yada", "class2" ),
      ( "Yada yada yada", "class2" ),
      ( "Hmmm yada hmmm blah hmmm", "class3" ),
      ( "Hmmm hmmm blur hmmm", "class3" ),
      ( "Hmmm hmmm yada hmmm", "class3" ),
      ( "Hmmm blop hmmm hmmm", "class3" ),
      ( "Hmmm hmmm blip hmmm", "class3" ),
      ( "Hmmm hmmm blap hmmm", "class3" )
    };

    private readonly TestCorpus _corpus = new( Documents );

    [Test]
    public void ClassificationTest() {
      var svm = MultiClassSvmTrainer.Train( _corpus.Documents, new WordCountKernel( _corpus ) );

      var testDoc = _corpus.Tokenize( "yada" );
      Assert.That( svm.Classify( testDoc ), Is.EqualTo( "class2" ) );

      testDoc = _corpus.Tokenize( "blah blah yada hmmm" );
      Assert.That( svm.Classify( testDoc ), Is.EqualTo( "class1" ) );

      testDoc = _corpus.Tokenize( "hmmm" );
      Assert.That( svm.Classify( testDoc ), Is.EqualTo( "class3" ) );
    }

    [Test]
    public void WordCountKernelTest() {
      var testDoc1 = _corpus.Tokenize( "yada blah" );
      var testDoc2 = _corpus.Tokenize( "blah yada" );
      Console.WriteLine( new WordCountKernel( _corpus ).Compute( testDoc1, testDoc2 ) );
    }
  }

  internal class TestCorpus : ICorpus {
    private int _currentWordId;
    private readonly Dictionary<string, int> _wordIds = new();
    private readonly Dictionary<int, int> _documentFrequencies = new();
    private readonly List<Observation<Document,string>> _documents = new();

    public TestCorpus( IEnumerable<(string doc, string label)> documents ) {
      foreach ( var (doc, label) in documents ) {
        AddObservation( doc, label );
      }
    }

    public IReadOnlyList<Observation<Document, string>> Documents => _documents;

    public int GetDocumentFrequency( int wordId ) {
      return _documentFrequencies.TryGetValue( wordId, out var freq ) ? freq : 0;
    }
    
    public int CorpusSize => Documents.Count;

    internal Document Tokenize( string text ) {
      var words = text.Split( " ", StringSplitOptions.RemoveEmptyEntries )
        .Select( s => s.ToLower() )
        .GroupBy( w => w )
        .Select( wf => new WordFrequency( MapWordId( wf.Key ), wf.Count() ) );
      return new Document( words );
    }

    private void AddObservation( string text, string label ) {
      var doc = Tokenize( text );
      foreach ( var wordFrequency in doc.Words ) {
        _documentFrequencies.TryGetValue( wordFrequency.WordId, out var curFreq );
        _documentFrequencies[wordFrequency.WordId] = curFreq + 1;
      }

      _documents.Add( new Observation<Document, string>( doc, label ) );
    }

    private int MapWordId( string word ) {
      if ( !_wordIds.ContainsKey( word ) ) {
        _wordIds[word] = _currentWordId++;
      }

      return _wordIds[word];
    }

  }
}
