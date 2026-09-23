# -*- coding: utf-8 -*-
"""
Module 3 - Chatbot de navigation - Définition des intents (v2, enrichie)

Cahier des charges, section 4.3 : "Analyse des parcours utilisateurs",
"Création d'intents et d'entités".

⚠️ CE FICHIER EST FAIT POUR ÊTRE MODIFIÉ. Version v2 : nombre d'exemples
par intent significativement augmenté (18-24 au lieu de 8-12) pour aider
le fine-tuning à converger — le premier essai (73 exemples au total) a
échoué à apprendre quoi que ce soit (loss bloquée au niveau du hasard).

Dès que vous avez accès aux vraies fonctionnalités/sections de Cheebo :
  1. Corrigez le champ "redirection" de chaque intent avec le vrai nom
     de section/écran de l'application (actuellement des placeholders)
  2. Ajoutez/supprimez des intents selon les vraies fonctionnalités
  3. Remplacez les phrases d'exemple par de VRAIES questions
     d'utilisateurs si vous y avez accès (logs de support, FAQ existante)
  4. Continuez à viser 20+ exemples par intent minimum
"""

INTENTS = {

    "ajouter_animal": {
        "description": "L'utilisateur veut ajouter un animal à son profil",
        "redirection": "Section: Mon Compte > Mes Animaux > Ajouter un animal",
        "reponse_type": (
            "Pour ajouter un animal : allez dans « Mon Compte », puis "
            "« Mes Animaux », et appuyez sur « Ajouter un animal ». "
            "Renseignez son nom, son espèce, sa race et sa date de naissance."
        ),
        "examples": [
            "Comment ajouter mon animal ?",
            "Comment enregistrer mon chien sur l'application ?",
            "Je veux ajouter un nouvel animal à mon profil",
            "Comment créer une fiche pour mon chat ?",
            "Où est-ce que j'ajoute mon animal de compagnie ?",
            "Comment inscrire mon lapin sur Cheebo ?",
            "Je n'arrive pas à ajouter mon chien, comment faire ?",
            "Comment enregistrer un nouvel animal ?",
            "Où puis-je déclarer mon chat sur l'app ?",
            "Comment faire pour ajouter un deuxième animal ?",
            "Je viens d'adopter un chien, comment l'ajouter sur Cheebo ?",
            "C'est où pour créer le profil de mon animal ?",
            "Comment associer mon chat à mon compte ?",
            "Je veux enregistrer mon nouveau chiot",
            "Comment ajouter une nouvelle fiche animal ?",
            "Est-ce que je peux ajouter plusieurs animaux ?",
            "Comment saisir les infos de mon animal ?",
            "Où déclarer un nouvel animal sur l'application ?",
            "Comment mettre mon chien dans l'application ?",
            "Je veux créer le profil de mon hamster",
        ],
    },

    "trouver_veterinaire": {
        "description": "L'utilisateur cherche un vétérinaire via l'app",
        "redirection": "Section: Vétérinaires > Recherche par localisation",
        "reponse_type": (
            "Pour trouver un vétérinaire : allez dans la section "
            "« Vétérinaires », puis autorisez la localisation ou entrez "
            "votre ville pour voir les cliniques les plus proches."
        ),
        "examples": [
            "Où trouver un vétérinaire ?",
            "Comment localiser une clinique vétérinaire proche ?",
            "Je cherche un vétérinaire dans ma ville",
            "Comment voir les vétérinaires disponibles sur l'app ?",
            "Où puis-je trouver un véto près de chez moi ?",
            "Y a-t-il une liste de vétérinaires sur Cheebo ?",
            "Comment prendre rendez-vous avec un vétérinaire ?",
            "Je veux trouver une clinique vétérinaire à Sousse",
            "Comment chercher un vétérinaire par région ?",
            "Où voir les avis sur les vétérinaires ?",
            "Comment contacter un vétérinaire via l'application ?",
            "Je ne trouve pas la liste des vétérinaires",
            "Comment trouver un véto ouvert maintenant ?",
            "Est-ce qu'il y a des vétérinaires près de Tunis ?",
            "Comment voir les cliniques vétérinaires disponibles ?",
            "Je veux le numéro d'un vétérinaire proche",
            "Comment filtrer les vétérinaires par spécialité ?",
            "Où consulter la carte des vétérinaires ?",
            "Comment savoir quel vétérinaire est le plus proche ?",
            "Je cherche une clinique pour urgence animale",
        ],
    },

    "identifiant_cheebo": {
        "description": "L'utilisateur veut comprendre l'identifiant Cheebo (numéro/QR code de son animal)",
        "redirection": "Section: Mes Animaux > Fiche animal > Identifiant Cheebo",
        "reponse_type": (
            "L'identifiant Cheebo est un code unique attribué à chaque "
            "animal enregistré, visible sur sa fiche. Il permet de "
            "retrouver son propriétaire en cas de perte. Vous le trouverez "
            "dans « Mes Animaux » > la fiche de votre animal."
        ),
        "examples": [
            "Comment fonctionne l'identifiant Cheebo ?",
            "Qu'est-ce que l'identifiant Cheebo ?",
            "À quoi sert mon numéro Cheebo ?",
            "Où trouver l'identifiant de mon animal ?",
            "C'est quoi le code Cheebo sur la fiche de mon chien ?",
            "Comment obtenir un identifiant pour mon chat ?",
            "Le QR code Cheebo, ça sert à quoi ?",
            "Comment imprimer l'identifiant de mon animal ?",
            "Mon animal a-t-il un identifiant unique ?",
            "Comment mettre à jour l'identifiant Cheebo ?",
            "À quoi correspond ce numéro sur la fiche de mon animal ?",
            "Comment fonctionne le système d'identification Cheebo ?",
            "Est-ce que l'identifiant Cheebo sert en cas de perte ?",
            "Comment scanner le QR code de mon animal ?",
            "Où est affiché l'identifiant unique de mon chat ?",
            "Comment relier l'identifiant Cheebo à mon numéro de téléphone ?",
            "Le numéro Cheebo est-il visible par les autres utilisateurs ?",
            "Comment vérifier l'identifiant d'un animal trouvé ?",
        ],
    },

    "creer_compte": {
        "description": "L'utilisateur veut créer un compte ou s'inscrire",
        "redirection": "Section: Inscription",
        "reponse_type": (
            "Pour créer un compte : appuyez sur « S'inscrire » depuis "
            "l'écran d'accueil, renseignez votre email et un mot de passe, "
            "puis confirmez votre adresse email."
        ),
        "examples": [
            "Comment créer un compte ?",
            "Comment m'inscrire sur Cheebo ?",
            "Je veux créer un compte utilisateur",
            "Comment faire pour s'inscrire sur l'application ?",
            "Où est le bouton d'inscription ?",
            "Comment créer mon profil Cheebo ?",
            "Je n'arrive pas à m'inscrire, aidez-moi",
            "Comment ouvrir un compte sur l'app ?",
            "Faut-il un email pour s'inscrire ?",
            "Comment valider mon inscription ?",
            "Est-ce que l'inscription est gratuite ?",
            "Comment s'enregistrer sur Cheebo avec Google ?",
            "Je veux m'inscrire avec mon numéro de téléphone",
            "Où puis-je créer un nouveau compte ?",
            "Comment finaliser la création de mon compte ?",
            "L'inscription prend combien de temps ?",
            "Comment recevoir le code de confirmation ?",
            "Je n'ai pas reçu l'email de confirmation, que faire ?",
        ],
    },

    "modifier_profil": {
        "description": "L'utilisateur veut modifier les infos de son profil ou de son animal",
        "redirection": "Section: Mon Compte > Modifier le profil",
        "reponse_type": (
            "Pour modifier votre profil ou celui de votre animal : allez "
            "dans « Mon Compte », sélectionnez la fiche à modifier et "
            "appuyez sur « Modifier »."
        ),
        "examples": [
            "Comment modifier les informations de mon animal ?",
            "Comment changer la photo de mon chien ?",
            "Je veux mettre à jour le profil de mon chat",
            "Comment modifier mon nom d'utilisateur ?",
            "Comment changer mon mot de passe ?",
            "Où modifier l'âge de mon animal ?",
            "Comment corriger une erreur sur la fiche de mon chien ?",
            "Je veux changer ma photo de profil",
            "Comment mettre à jour mes informations personnelles ?",
            "Comment modifier l'adresse associée à mon compte ?",
            "Comment changer le poids de mon animal sur sa fiche ?",
            "Je veux corriger la race de mon chat",
            "Comment mettre à jour mon numéro de téléphone ?",
            "Où éditer les informations de mon compte ?",
            "Comment changer mon adresse email ?",
            "Je veux actualiser la fiche de mon animal",
            "Comment renommer mon animal sur l'app ?",
        ],
    },

    "gerer_notifications": {
        "description": "L'utilisateur veut gérer ses notifications",
        "redirection": "Section: Paramètres > Notifications",
        "reponse_type": (
            "Pour gérer vos notifications : allez dans « Paramètres », "
            "puis « Notifications », et activez/désactivez les catégories "
            "souhaitées."
        ),
        "examples": [
            "Comment désactiver les notifications ?",
            "Je reçois trop de notifications, comment les réduire ?",
            "Comment activer les notifications push ?",
            "Où gérer mes préférences de notification ?",
            "Comment ne plus recevoir d'emails de Cheebo ?",
            "Je ne reçois aucune notification, pourquoi ?",
            "Comment couper les notifications de la communauté ?",
            "Où sont les paramètres de notification ?",
            "Comment arrêter les notifications le soir ?",
            "Je veux seulement les notifications importantes",
            "Comment activer les alertes vétérinaires ?",
            "Pourquoi je reçois des notifications en double ?",
            "Comment paramétrer les alertes de rappel de vaccin ?",
            "Je veux désactiver les notifications de la communauté seulement",
        ],
    },

    "utiliser_doctobot": {
        "description": "L'utilisateur veut savoir comment utiliser le chatbot vétérinaire Doctobot",
        "redirection": "Section: Doctobot (chat)",
        "reponse_type": (
            "Pour parler à Doctobot : appuyez sur l'icône de chat en bas "
            "de l'écran, puis décrivez les symptômes de votre animal. "
            "Doctobot vous donnera des conseils et vous orientera vers un "
            "vétérinaire si nécessaire."
        ),
        "examples": [
            "Comment parler à Doctobot ?",
            "Comment poser une question sur la santé de mon animal ?",
            "Où trouver le chatbot vétérinaire ?",
            "Comment utiliser l'assistant santé animale ?",
            "Doctobot, c'est quoi et comment ça marche ?",
            "Comment décrire les symptômes de mon chien à Doctobot ?",
            "Où puis-je discuter avec l'assistant vétérinaire ?",
            "Comment accéder au chat de conseils santé ?",
            "Comment démarrer une conversation avec Doctobot ?",
            "Doctobot peut-il donner un diagnostic ?",
            "Comment ouvrir le chatbot de santé animale ?",
            "Est-ce que Doctobot répond en français ?",
            "Comment quitter une conversation avec Doctobot ?",
            "À quoi sert Doctobot exactement ?",
        ],
    },

    "supprimer_compte": {
        "description": "L'utilisateur veut supprimer son compte",
        "redirection": "Section: Paramètres > Confidentialité > Supprimer le compte",
        "reponse_type": (
            "Pour supprimer votre compte : allez dans « Paramètres », "
            "« Confidentialité », puis « Supprimer mon compte ». Cette "
            "action est définitive et supprime toutes vos données."
        ),
        "examples": [
            "Comment supprimer mon compte ?",
            "Je veux fermer mon compte Cheebo",
            "Comment se désinscrire de l'application ?",
            "Comment effacer définitivement mon profil ?",
            "Je ne veux plus utiliser Cheebo, comment supprimer mes données ?",
            "Comment résilier mon compte ?",
            "Est-ce que la suppression du compte est définitive ?",
            "Comment supprimer toutes mes données personnelles ?",
            "Je veux quitter la plateforme définitivement",
            "Comment fermer mon compte et celui de mes animaux ?",
            "Combien de temps pour supprimer un compte ?",
            "Puis-je récupérer mon compte après suppression ?",
        ],
    },

    "contacter_support": {
        "description": "L'utilisateur a un problème technique ou veut contacter le support",
        "redirection": "Section: Aide > Contacter le support",
        "reponse_type": (
            "Pour contacter le support : allez dans « Aide », puis "
            "« Contacter le support », et décrivez votre problème. Vous "
            "recevrez une réponse par email sous 48h."
        ),
        "examples": [
            "Comment contacter le support ?",
            "J'ai un problème technique, qui contacter ?",
            "L'application plante, comment signaler le bug ?",
            "Comment joindre le service client ?",
            "Où envoyer une réclamation ?",
            "Comment signaler un problème sur l'app ?",
            "Y a-t-il un numéro d'assistance ?",
            "Comment obtenir de l'aide rapidement ?",
            "L'application ne fonctionne pas, que faire ?",
            "Comment envoyer un message à l'équipe Cheebo ?",
            "Où signaler un bug technique ?",
            "Comment avoir une réponse du support Cheebo ?",
            "L'app se ferme toute seule, comment le signaler ?",
            "Comment demander de l'aide sur l'application ?",
        ],
    },

    "hors_sujet": {
        "description": "Message qui ne concerne pas la navigation dans l'app (fourre-tout, à rediriger ailleurs ou ignorer)",
        "redirection": None,
        "reponse_type": (
            "Je suis l'assistant de navigation Cheebo, je peux vous aider "
            "à utiliser l'application. Pour des questions sur la santé de "
            "votre animal, parlez plutôt à Doctobot !"
        ),
        "examples": [
            "Bonjour, comment ça va ?",
            "Merci beaucoup",
            "Quel temps fait-il aujourd'hui ?",
            "Mon chat a des puces, que faire ?",
            "Mon chien vomit depuis ce matin",
            "Raconte-moi une blague",
            "C'est qui le président de la Tunisie ?",
            "Je m'ennuie",
            "Combien coûte un abonnement premium ?",
            "Quelle est la capitale de la France ?",
            "Mon chien a du mal à respirer",
            "Peux-tu m'aider avec mes devoirs ?",
            "Quelle heure est-il ?",
            "Mon chat ne mange plus depuis 2 jours",
        ],
    },
}
