"""Independent scaled real-SOS oracle, adapted from project research T04.

Apache-2.0 project code. No production modal routines are used. Both propagation
checks share Lyapunov stationary initialization; residuals alone do not certify
forward accuracy. This is a numerical proxy, not measured acoustics.
"""
import warnings
import numpy as np
from scipy import linalg, signal


def sos_state_space(sos, *, balance=True, return_transform=False):
    """Probe the implemented SOS one-step transition on real basis states."""
    sos=np.asarray(sos,float)
    n=2*len(sos)
    F=np.empty((n,n)); C=np.empty(n)
    for i in range(n):
        zi=np.zeros(n); zi[i]=1.
        y,final=signal.sosfilt(sos,[0.],zi=zi.reshape(-1,2))
        F[:,i]=final.ravel(); C[i]=y[0]
    y,final=signal.sosfilt(sos,[1.],zi=np.zeros((len(sos),2)))
    G=final.ravel(); D=float(y[0])
    T=np.eye(n)
    if balance:
        # A diagonal similarity only, never a modal eigendecomposition.
        F,T=linalg.matrix_balance(F,permute=False)
        G=linalg.solve(T,G)
        C=C@T
        # Equilibrate state variances using finite controllability diagonals.
        # This is just diagonal coordinate scaling, not approximate covariance
        # initialization: the full stationary covariance is solved afterwards.
        diagonal=np.zeros(n); impulse=G.copy()
        for _ in range(4096):
            diagonal+=impulse*impulse
            impulse=F@impulse
        scale=np.sqrt(np.maximum(diagonal,np.finfo(float).tiny))
        F=F*scale[None,:]/scale[:,None]
        G=G/scale
        C=C*scale
        T=T@np.diag(scale)
    return (F,G,C,D,T) if return_transform else (F,G,C,D)


def stationary_covariance(F,G):
    Q=np.outer(G,G)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        P=linalg.solve_discrete_lyapunov(F,Q,method='bilinear')
        direct=linalg.solve_discrete_lyapunov(F,Q,method='direct')
    asymmetry=float(linalg.norm(P-P.T)/max(linalg.norm(P),1e-300))
    P=(P+P.T)/2
    norm_res=float(linalg.norm(F@P@F.T-P+Q)/max(linalg.norm(P)+linalg.norm(Q),1e-300))
    chol=linalg.cholesky(P,lower=True)
    return P,chol,{'stationary_normalized_residual':norm_res,
                   'stationary_asymmetry_before_symmetrization':asymmetry,
                   'stationary_covariance_condition':float(np.linalg.cond(P)),
                   'stationary_direct_vs_bilinear_norm_relative':float(linalg.norm(direct-P)/max(linalg.norm(P),1e-300)),
                   'stationary_warnings':[str(w.message) for w in caught]}


def real_expected_filtered_power(synthesis_sos,analysis_sos,envelope,*,square_root=True):
    """Propagate the joint real state covariance of y=A*(e*(B*w))."""
    e=np.asarray(envelope,float)
    if e.ndim!=1 or not np.isfinite(e).all():
        raise ValueError('finite vector required')
    Fb,Gb,Cb,Db=sos_state_space(synthesis_sos)
    Fa,Ga,Ca,Da=sos_state_space(analysis_sos)
    Pb,Lb,diag=stationary_covariance(Fb,Gb)
    nb,na=len(Fb),len(Fa); n=nb+na
    M=linalg.block_diag(Fb,Fa)
    g=np.r_[Gb,np.zeros(na)]
    h=np.r_[np.zeros(nb),Ca]
    P=np.zeros((n,n)); P[:nb,:nb]=Pb
    S=np.zeros((n,n)); S[:nb,:nb]=Lb
    out=np.empty(len(e))
    lower=np.outer(Ga,Cb)
    for k,ek in enumerate(e):
        M[nb:,:nb]=ek*lower
        g[nb:]=ek*Ga*Db
        h[:nb]=Da*ek*Cb
        j=Da*ek*Db
        if square_root:
            projection=h@S
            out[k]=projection@projection+j*j
            # S' S'^T = M S S^T M^T + g g^T; no covariance subtraction.
            block=np.column_stack((M@S,g))
            S=np.linalg.qr(block.T,mode='r').T
        else:
            out[k]=h@P@h+j*j
            P=M@P@M.T+np.outer(g,g)
            P=(P+P.T)/2
    diag.update({'synthesis_state_count':nb,'analysis_state_count':na,
                 'synthesis_variance':float(Cb@Pb@Cb+Db*Db),
                 'minimum_raw_power':float(out.min(initial=0.)),
                 'method':'real SOS square-root QR' if square_root else 'real SOS covariance'})
    return out,diag


def direct_sos_factor_power(synthesis_sos,analysis_sos,envelope):
    """O(N squared) deterministic impulse-column oracle using only sosfilt.

    Each column is one independent unit-variance random input: initial synthesis
    Cholesky factor columns, then the finite white-innovation identity matrix.
    Sum of squared output coefficients equals exact expected output power.
    This independently checks QR propagation while sharing only stationarity.
    """
    e=np.asarray(envelope,float); N=len(e)
    F,G,C,D,T=sos_state_space(synthesis_sos,return_transform=True)
    _,L,_=stationary_covariance(F,G)
    nb=len(F)
    W=np.zeros((N,nb+N)); W[np.arange(N),nb+np.arange(N)]=1.
    zi=np.zeros((len(synthesis_sos),2,nb+N))
    zi[:,:,:nb]=(T@L).reshape(len(synthesis_sos),2,nb)
    source,_=signal.sosfilt(synthesis_sos,W,axis=0,zi=zi)
    source*=e[:,None]
    output=signal.sosfilt(analysis_sos,source,axis=0)
    return np.einsum('ij,ij->i',output,output)


