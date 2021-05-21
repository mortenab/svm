using System;
using System.Linq;

namespace Svm.Binary {
  /// <summary>
  /// Based on the pseudo code here:
  /// https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
  /// </summary>
  internal class SequentialMinimalOptimization<TValue> {
    private readonly BinaryObservation<TValue>[] _trainingSet;
    private readonly double[] _alphas;
    private readonly IKernel<TValue> _k;
    private readonly Random _rand;
    private readonly double[] _errorCache;
    private double _b;
    private const double Tolerance = 10E-3;
    private const double Eps = 2.2204460492503131e-16;
    private const double C = 0.05;
    private int _y2;
    private double _e2;
    private double _alpha2;

    public SequentialMinimalOptimization( BinaryObservation<TValue>[] trainingSet, IKernel<TValue> k ) {
      _k = k;
      _trainingSet = trainingSet;
      _alphas = new double[_trainingSet.Length];
      _errorCache = new double[trainingSet.Length];
      _rand = new Random( 42 );
    }

    private int TakeStep( int i1, int i2 ) {
      if ( i1 == i2 ) {
        return 0;
      }

      double alpha1 = _alphas[i1];
      int y1 = _trainingSet[i1].Label;
      double E1;

      if ( alpha1 > 0 && alpha1 < C ) {
        E1 = _errorCache[i1];
      }
      else {
        E1 = SvmOutput( i1 ) - y1;
      }

      int s = y1 * _y2;
      double L, H;
      if ( s == 1 ) {
        L = Math.Max( 0, alpha1 + _alpha2 - C );
        H = Math.Min( C, alpha1 + _alpha2 );
      }
      else {
        L = Math.Max( 0, _alpha2 - alpha1 );
        H = Math.Min( C, C + _alpha2 - alpha1 );
      }

      if ( Math.Abs(L - H) < 10E-9 ) {
        return 0;
      }

      double k11 = _k.Compute( _trainingSet[i1].Value, _trainingSet[i1].Value );
      double k12 = _k.Compute( _trainingSet[i1].Value, _trainingSet[i2].Value );
      double k22 = _k.Compute( _trainingSet[i2].Value, _trainingSet[i2].Value );

      double eta = 2 * k12 - k11 - k22;
      double newAlpha2;
      double Lobj, Hobj;

      if ( eta < 0 ) {
        newAlpha2 = _alpha2 - _y2 * ( E1 - _e2 ) / eta;
        if ( newAlpha2 < L )
          newAlpha2 = L;
        else if ( newAlpha2 > H )
          newAlpha2 = H;
      }
      else {
        double c1 = eta / 2;
        double c2 = _y2 * ( E1 - _e2 ) - eta * _alpha2;
        Lobj = c1 * L * L + c2 * L;
        Hobj = c1 * H * H + c2 * H;

        if ( Lobj > Hobj + Eps )
          newAlpha2 = L;
        else if ( Lobj < Hobj - Eps )
          newAlpha2 = H;
        else
          newAlpha2 = _alpha2;
      }

      if ( Math.Abs( newAlpha2 - _alpha2 ) < Eps * ( newAlpha2 + _alpha2 + Eps ) ) {
        return 0;
      }

      var newAlpha1 = alpha1 - s * ( newAlpha2 - _alpha2 );
      if ( newAlpha1 < 0 ) {
        newAlpha2 += s * newAlpha1;
        newAlpha1 = 0;
      }
      else if ( newAlpha1 > C ) {
        newAlpha2 += s * ( newAlpha1 - C );
        newAlpha1 = C;
      }

      double bNew;
      if ( newAlpha1 > 0 && newAlpha1 < C ) {
        bNew = _b + E1 + y1 * ( newAlpha1 - alpha1 ) * k11 + _y2 * ( newAlpha2 - _alpha2 ) * k12;
      }
      else {
        if ( newAlpha2 > 0 && newAlpha2 < C )
          bNew = _b + _e2 + y1 * ( newAlpha1 - alpha1 ) * k12 + _y2 * ( newAlpha2 - _alpha2 ) * k22;
        else {
          var b1 = _b + E1 + y1 * ( newAlpha1 - alpha1 ) * k11 + _y2 * ( newAlpha2 - _alpha2 ) * k12;
          var b2 = _b + _e2 + y1 * ( newAlpha1 - alpha1 ) * k12 + _y2 * ( newAlpha2 - _alpha2 ) * k22;
          bNew = ( b1 + b2 ) / 2;
        }
      }

      var deltaB = _b - bNew;
      _b = bNew;

      double t1 = y1 * ( newAlpha1 - alpha1 );
      double t2 = _y2 * ( newAlpha2 - _alpha2 );
      for ( int i = 0; i < _errorCache.Length; i++ ) {
        if ( _alphas[i] > 0 && _alphas[i] < C ) {
          _errorCache[i] += t1 * _k.Compute( _trainingSet[i1].Value, _trainingSet[i].Value ) +
                            t2 * _k.Compute( _trainingSet[i2].Value, _trainingSet[i].Value ) + deltaB;
        }
      }

      _errorCache[i1] = _errorCache[i2] = 0;
      _alphas[i1] = newAlpha1;
      _alphas[i2] = newAlpha2;
      return 1;
    }

    private int ExamineExample( int i2 ) {
      double r2;
      int i1, j;
      _y2 = _trainingSet[i2].Label;
      _alpha2 = _alphas[i2];


      if ( _alpha2 > 0 && _alpha2 < C ) {
        _e2 = _errorCache[i2];
      }
      else {
        _e2 = SvmOutput( i2 ) - _y2;
      }

      r2 = _e2 * _y2;

      if ( ( r2 < -Tolerance && _alpha2 < C ) || ( r2 > Tolerance && _alpha2 > 0 ) ) {
        i1 = -1;
        double max = 0;

        for ( j = 0; j < _alphas.Length; j++ ) {
          if ( _alphas[j] > 0 && _alphas[j] < C ) {
            var E1 = _errorCache[j];
            var temp = Math.Abs( E1 - _e2 );
            if ( temp > max ) {
              i1 = j;
              max = temp;
            }
          }
        }

        if ( i1 >= 0 )
          if ( TakeStep( i1, i2 ) == 1 )
            return 1;

        int r = _rand.Next( _alphas.Length );

        for ( j = r; j < _alphas.Length + r; j++ ) {
          i1 = j % _alphas.Length;
          if ( _alphas[i1] > 0 && _alphas[i1] < C )
            if ( TakeStep( i1, i2 ) == 1 ) {
              return 1;
            }
        }

        r = _rand.Next( _alphas.Length );
        for ( j = r; j < _alphas.Length + r; j++ ) {
          i1 = j % _alphas.Length;
          if ( TakeStep( i1, i2 ) == 1 )
            return 1;
        }
      }

      return 0;
    }

    private double SvmOutput( int index ) {
      double s = 0;
      for ( int i = 0; i < _alphas.Length; i++ ) {
        if ( _alphas[i] > 0 ) {
          s += _alphas[i] * _trainingSet[i].Label * _k.Compute( _trainingSet[i].Value, _trainingSet[index].Value );
        }
      }

      s -= _b;

      return s;
    }

    public BinarySvm<TValue> Optimize() {
      int numChanged = 0;
      bool examineAll = true;

      while ( numChanged > 0 || examineAll ) {
        numChanged = 0;

        if ( examineAll ) {
          for ( int i = 0; i < _trainingSet.Length; i++ ) {
            numChanged += ExamineExample( i );
          }
        }
        else {
          for ( int i = 0; i < _trainingSet.Length; i++ ) {
            if ( _alphas[i] > 0 && _alphas[i] < C ) {
              numChanged += ExamineExample( i );
            }
          }
        }

        if ( examineAll ) {
          examineAll = false;
        }
        else if ( numChanged == 0 ) {
          examineAll = true;
        }
      }

      var supportVectors = _alphas.Zip( _trainingSet )
        .Where( t => t.First > 0 )
        .Select( t => new SupportVector<TValue>( t.First, t.Second ) )
        .ToList();

      return new BinarySvm<TValue>( _b, supportVectors, _k );
    }
  }
}
